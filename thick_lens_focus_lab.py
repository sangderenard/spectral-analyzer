"""Forward-only C++ lens bench with orthographic light views.

Light transport, spectral integration, and spectral->RGB conversion are all
performed through _spectral_kernels.RayTracer.
"""

from __future__ import annotations

import colorsys
import concurrent.futures
import math
import os
import queue
import sys
import threading
import ctypes
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Directory containing the GLSL compute shaders (ray_bvh_intersect.comp.glsl etc.)
_SHADER_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "shaders")

try:
    import pygame
except Exception as exc:
    raise RuntimeError("pygame is required. Install with: pip install pygame") from exc

import _spectral_kernels as _sk
from base_gl_renderer import BaseGLRenderer
from material_db import MAX_SPECTRAL_BANDS, MaterialDatabase
from sensor_film_db import MAX_SENSOR_FILM_SLOTS, SensorFilmDatabase
from spectral_material import Material, RadianceProfile, SpectralBand

try:
    import torch
except Exception:
    torch = None

EPS = 1.0e-9
C_LIGHT = 299_792_458.0
DEFAULT_FREQ_HZ = (C_LIGHT / np.linspace(700e-9, 380e-9, MAX_SPECTRAL_BANDS)).astype(np.float64)
DEBUG_BDPT_BACKTRACE_ONLY_DEFAULT = False
UV_PAGE_RES_DEFAULT = 512
UV_HOT_GROUP_LIMIT_DEFAULT = 8
UV_HDR_CHANNELS = 11


class FrameProfiler:
    """Small rolling profiler for Python-side frame and display costs."""

    def __init__(self, report_every: int = 60) -> None:
        self.report_every = max(1, int(report_every))
        self.frame = 0
        self._order: List[str] = []
        self._data: Dict[str, List[float]] = {}
        self._t0: Dict[str, float] = {}

    def begin(self, name: str) -> None:
        if name not in self._data:
            self._data[name] = []
            self._order.append(name)
        self._t0[name] = time.perf_counter()

    def end(self, name: str) -> None:
        t0 = self._t0.pop(name, None)
        if t0 is None:
            return
        self._data.setdefault(name, []).append((time.perf_counter() - t0) * 1.0e3)

    def measure(self, name: str):
        profiler = self

        class _Scope:
            def __enter__(self_inner):
                profiler.begin(name)
                return self_inner

            def __exit__(self_inner, _exc_type, _exc, _tb):
                profiler.end(name)
                return False

        return _Scope()

    def tick(self, extra: str = "") -> None:
        self.frame += 1
        if self.frame % self.report_every:
            return
        parts = []
        for key in self._order:
            vals = self._data.get(key, [])
            if vals:
                parts.append(f"{key}={float(np.mean(vals)):.2f}ms")
        suffix = f"  {extra}" if extra else ""
        print(f"[py-profile f={self.frame}] " + "  ".join(parts) + suffix, flush=True)
        for vals in self._data.values():
            vals.clear()


def _nm_to_hz(nm: np.ndarray) -> np.ndarray:
    nm = np.asarray(nm, dtype=np.float64)
    nm = np.clip(nm, 1.0, None)
    return C_LIGHT / (nm * 1.0e-9)


def _freq_to_hue_rgb(freq_hz: float) -> Tuple[int, int, int]:
    wl_nm = float(np.clip(C_LIGHT / max(float(freq_hz), EPS) * 1.0e9, 380.0, 700.0))
    t = (wl_nm - 380.0) / (700.0 - 380.0)
    hue = 0.75 * (1.0 - t)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    return int(255.0 * r), int(255.0 * g), int(255.0 * b)


def _wavelength_to_rgb_weights(wl_nm: np.ndarray) -> np.ndarray:
    wl = np.asarray(wl_nm, dtype=np.float64)
    r = np.zeros_like(wl, dtype=np.float64)
    g = np.zeros_like(wl, dtype=np.float64)
    b = np.zeros_like(wl, dtype=np.float64)

    m = (wl >= 380.0) & (wl < 440.0)
    r[m] = -(wl[m] - 440.0) / (440.0 - 380.0)
    b[m] = 1.0

    m = (wl >= 440.0) & (wl < 490.0)
    g[m] = (wl[m] - 440.0) / (490.0 - 440.0)
    b[m] = 1.0

    m = (wl >= 490.0) & (wl < 510.0)
    g[m] = 1.0
    b[m] = -(wl[m] - 510.0) / (510.0 - 490.0)

    m = (wl >= 510.0) & (wl < 580.0)
    r[m] = (wl[m] - 510.0) / (580.0 - 510.0)
    g[m] = 1.0

    m = (wl >= 580.0) & (wl < 645.0)
    r[m] = 1.0
    g[m] = -(wl[m] - 645.0) / (645.0 - 580.0)

    m = (wl >= 645.0) & (wl <= 700.0)
    r[m] = 1.0

    edge = np.ones_like(wl, dtype=np.float64)
    m = (wl >= 380.0) & (wl < 420.0)
    edge[m] = 0.3 + 0.7 * (wl[m] - 380.0) / (420.0 - 380.0)
    m = (wl > 645.0) & (wl <= 700.0)
    edge[m] = 0.3 + 0.7 * (700.0 - wl[m]) / (700.0 - 645.0)

    out = np.column_stack([r * edge, g * edge, b * edge])
    return np.ascontiguousarray(out, dtype=np.float64)


def _sensor_rgb_sensitivity_bands(freq_hz: np.ndarray) -> np.ndarray:
    """Return RGB sensor emission spectra as (3, n_bands), normalized per row."""
    f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    if f.size <= 0:
        return np.zeros((3, 0), dtype=np.float64)
    wl = np.clip(C_LIGHT / np.maximum(f, EPS) * 1.0e9, 360.0, 760.0)

    def gaussian(center_nm: float, sigma_nm: float) -> np.ndarray:
        return np.exp(-0.5 * ((wl - center_nm) / max(sigma_nm, EPS)) ** 2)

    # Broad, overlapping camera-like sensitivity lobes.  These are launch
    # spectra for reverse/sensor paths, not display colors.
    r = gaussian(610.0, 42.0)
    g = gaussian(540.0, 38.0)
    b = gaussian(460.0, 32.0)
    curves = np.stack([r, g, b], axis=0)
    curves /= np.maximum(curves.max(axis=1, keepdims=True), 1.0e-12)
    return np.ascontiguousarray(curves, dtype=np.float64)


def _free_frequency_hits_to_rgb(
    pixel_idx: np.ndarray,
    freq_idx: np.ndarray,
    power: np.ndarray,
    n_px: int,
    freq_hz: np.ndarray,
) -> np.ndarray:
    n = int(max(1, n_px))
    if pixel_idx.size <= 0 or freq_idx.size <= 0 or power.size <= 0:
        return np.zeros((n, n, 3), dtype=np.float32)

    f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    if f.size <= 0:
        return np.zeros((n, n, 3), dtype=np.float32)

    wl_nm = np.clip(C_LIGHT / np.maximum(f, EPS) * 1.0e9, 380.0, 700.0)
    w_rgb = _wavelength_to_rgb_weights(wl_nm)

    rgb_lin = np.zeros((n * n, 3), dtype=np.float64)
    pr = np.asarray(power, dtype=np.float64)
    fi = np.asarray(freq_idx, dtype=np.int32)
    pi = np.asarray(pixel_idx, dtype=np.int32)

    np.add.at(rgb_lin[:, 0], pi, pr * w_rgb[fi, 0])
    np.add.at(rgb_lin[:, 1], pi, pr * w_rgb[fi, 1])
    np.add.at(rgb_lin[:, 2], pi, pr * w_rgb[fi, 2])
    rgb = rgb_lin.reshape(n, n, 3)

    white = float(np.percentile(rgb, 99.8)) if rgb.size else 0.0
    white = max(white, 1.0e-8)
    y = np.log1p((rgb / white) * 6.0) / np.log1p(6.0)
    y = y / (1.0 + 0.18 * y)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _spectral_image_to_rgb(image_b_h_w: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    p = np.maximum(0.0, np.asarray(image_b_h_w, dtype=np.float64))
    f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    if p.ndim != 3 or p.shape[0] <= 0:
        h = int(p.shape[1]) if p.ndim >= 2 else 1
        w = int(p.shape[2]) if p.ndim >= 3 else 1
        return np.zeros((h, w, 3), dtype=np.float32)
    n_b = int(min(p.shape[0], f.size))
    if n_b <= 0:
        return np.zeros((p.shape[1], p.shape[2], 3), dtype=np.float32)

    wl_nm = np.clip(C_LIGHT / np.maximum(f[:n_b], EPS) * 1.0e9, 380.0, 700.0)
    w_rgb = _wavelength_to_rgb_weights(wl_nm)
    rgb = np.tensordot(p[:n_b], w_rgb, axes=(0, 0))

    white = float(np.percentile(rgb, 99.8)) if rgb.size else 0.0
    white = max(white, 1.0e-8)
    y = np.log1p((rgb / white) * 6.0) / np.log1p(6.0)
    y = y / (1.0 + 0.18 * y)
    luma = 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
    y = 0.90 * y + 0.10 * luma[..., None]
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _sensor_integral_to_display_rgb(photons_per_pixel: np.ndarray, electrons_per_pixel: np.ndarray) -> np.ndarray:
    photons = np.maximum(0.0, np.asarray(photons_per_pixel, dtype=np.float32))
    electrons = np.maximum(0.0, np.asarray(electrons_per_pixel, dtype=np.float32))

    white = float(np.percentile(photons, 99.5)) if photons.size else 1.0
    white = max(white, 1.0e-8)
    y = np.log1p((photons / white) * 6.0) / np.log1p(6.0)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)

    snr_proxy = np.sqrt(electrons)
    snr_white = float(np.percentile(snr_proxy, 99.0)) if snr_proxy.size else 1.0
    snr_white = max(snr_white, 1.0e-8)
    tint = np.clip(snr_proxy / snr_white, 0.0, 1.0).astype(np.float32)
    r = y
    g = np.clip(y * (0.92 + 0.08 * tint), 0.0, 1.0)
    b = np.clip(y * (0.86 + 0.14 * tint), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _sidecar_bandwidths(freq_hz: np.ndarray) -> np.ndarray:
    freq_hz = np.ascontiguousarray(np.asarray(freq_hz, dtype=np.float64).reshape(-1), dtype=np.float64)
    n = int(freq_hz.size)
    if n <= 1:
        base = float(freq_hz[0]) * 0.25 if n == 1 else 1.0
        return np.full((max(1, n),), max(base, 1.0), dtype=np.float64)
    d = np.abs(np.diff(freq_hz))
    left = np.empty((n,), dtype=np.float64)
    right = np.empty((n,), dtype=np.float64)
    left[0] = d[0]
    left[1:] = d
    right[:-1] = d
    right[-1] = d[-1]
    bw = np.maximum(0.5 * np.minimum(left, right), 1.0)
    return np.ascontiguousarray(bw, dtype=np.float64)


def _make_sidecar_spectral_bands(
    sidecar: FreeFrequencySidecar,
    *,
    reflectance: float,
    transmittance: float,
    diffuse_frac: float,
    emission_scale: float,
    ior_real: float,
    ior_imag: float,
) -> List[SpectralBand]:
    freq_hz = sidecar.freq_hz
    bw_hz = _sidecar_bandwidths(freq_hz)
    weight = np.ascontiguousarray(np.asarray(sidecar.weight, dtype=np.float64).reshape(-1), dtype=np.float64)
    if weight.size != freq_hz.size:
        weight = np.ones_like(freq_hz, dtype=np.float64)
    bands: List[SpectralBand] = []
    for freq, bw, wt in zip(freq_hz, bw_hz, weight):
        bands.append(
            SpectralBand(
                center_hz=float(freq),
                bandwidth_hz=float(bw),
                reflectance=float(reflectance),
                transmittance=float(transmittance),
                diffuse_frac=float(diffuse_frac),
                emission=float(emission_scale) * float(wt),
                reemission=0.0,
                ior_real=float(ior_real),
                ior_imag=float(ior_imag),
            )
        )
    return bands


def _make_red_only_spectral_bands(
    sidecar: FreeFrequencySidecar,
    *,
    emission_scale: float = 1.0,
    red_cutoff_nm: float = 600.0,
) -> List[SpectralBand]:
    """Spectral bands that emit only at wavelengths longer than red_cutoff_nm."""
    freq_hz = sidecar.freq_hz
    bw_hz   = _sidecar_bandwidths(freq_hz)
    bands: List[SpectralBand] = []
    for freq, bw in zip(freq_hz, bw_hz):
        wl_nm = (C_LIGHT / max(freq, EPS)) * 1.0e9
        em = float(emission_scale) if wl_nm >= red_cutoff_nm else 0.0
        bands.append(SpectralBand(
            center_hz=float(freq),
            bandwidth_hz=float(bw),
            reflectance=0.0,
            transmittance=0.0,
            diffuse_frac=0.0,
            emission=em,
            reemission=0.0,
            ior_real=1.0,
            ior_imag=0.0,
        ))
    return bands


def _make_dispersive_lens_bands(
    sidecar: FreeFrequencySidecar,
    *,
    base_ior: float,
    reflectance: float = 0.02,
    transmittance: float = 0.98,
) -> List[SpectralBand]:
    # Keep lens IOR spectrally constant here.  The previous Python-side Cauchy
    # term injected artificial chromatic fringing in this lab path.
    ior = np.full_like(np.asarray(sidecar.wavelength_nm, dtype=np.float64), float(base_ior), dtype=np.float64)
    bands: List[SpectralBand] = []
    freq_hz = sidecar.freq_hz
    bw_hz = _sidecar_bandwidths(freq_hz)
    for freq, bw, n_re in zip(freq_hz, bw_hz, ior):
        bands.append(
            SpectralBand(
                center_hz=float(freq),
                bandwidth_hz=float(bw),
                reflectance=float(reflectance),
                transmittance=float(transmittance),
                diffuse_frac=0.0,
                emission=0.0,
                reemission=0.0,
                ior_real=float(n_re),
                ior_imag=0.0,
            )
        )
    return bands


def _orient_surface_patch_outward(
    tri_arr: np.ndarray,
    tri_ids: Sequence[int],
    expected_x_sign: float,
) -> None:
    idx = np.asarray(tri_ids, dtype=np.int32)
    if idx.size <= 0:
        return
    norms = np.stack([_normal(t[0], t[1], t[2]) for t in tri_arr[idx]], axis=0)
    mean_x = float(np.mean(norms[:, 0]))
    if mean_x * float(expected_x_sign) < 0.0:
        tri_arr[idx] = tri_arr[idx][:, [0, 2, 1], :]


@dataclass
class FreeFrequencySidecar:
    """Representative free-frequency table for spectral bands.

    Stores wavelength/frequency/weight triplets used to drive the fixed-width
    vectorized band path in C++ while keeping physically meaningful sampling.
    """

    wavelength_nm: np.ndarray
    weight: np.ndarray

    @classmethod
    def from_prepared(
        cls,
        wavelength_nm: np.ndarray,
        weight: np.ndarray | None = None,
    ) -> "FreeFrequencySidecar":
        wl = np.ascontiguousarray(np.asarray(wavelength_nm, dtype=np.float64).reshape(-1), dtype=np.float64)
        if wl.size <= 0:
            raise ValueError("wavelength_nm must contain at least one entry")
        if weight is None:
            wt = np.ones_like(wl, dtype=np.float64)
        else:
            wt = np.ascontiguousarray(np.asarray(weight, dtype=np.float64).reshape(-1), dtype=np.float64)
            if wt.size != wl.size:
                raise ValueError("weight must have the same length as wavelength_nm")
        wt = np.clip(wt, 0.0, None)
        sw = float(np.sum(wt))
        if sw <= EPS:
            wt = np.ones_like(wl, dtype=np.float64)
            sw = float(wl.size)
        wt /= sw
        return cls(wavelength_nm=wl, weight=wt)

    @classmethod
    def lazy_prepare(
        cls,
        n_bands: int,
        wl_min_nm: float = 380.0,
        wl_max_nm: float = 700.0,
    ) -> "FreeFrequencySidecar":
        n = int(max(1, n_bands))
        anchors = np.array([390.0, 420.0, 460.0, 500.0, 540.0, 580.0, 620.0, 670.0], dtype=np.float64)
        if n == anchors.size:
            wl = anchors
        else:
            q = np.linspace(0.0, 1.0, n, dtype=np.float64)
            qa = np.linspace(0.0, 1.0, anchors.size, dtype=np.float64)
            wl = np.interp(q, qa, anchors)
        wl = np.clip(wl, min(wl_min_nm, wl_max_nm), max(wl_min_nm, wl_max_nm))
        center = 555.0
        sigma = 85.0
        wt = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
        return cls.from_prepared(wl, wt)

    @property
    def freq_hz(self) -> np.ndarray:
        return np.ascontiguousarray(_nm_to_hz(self.wavelength_nm), dtype=np.float64)

    @property
    def table(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack([self.wavelength_nm, self.freq_hz, self.weight]),
            dtype=np.float64,
        )


@dataclass
class LensConfig:
    center_x: float
    thickness: float
    aperture_radius: float
    radius_front: float
    radius_back: float
    ior: float

    @property
    def x_front(self) -> float:
        return self.center_x - 0.5 * self.thickness

    @property
    def x_back(self) -> float:
        return self.center_x + 0.5 * self.thickness


@dataclass
class ObjectPlaneConfig:
    x: float = 0.08
    radius: float = 0.050
    pixels: int = 17
    directivity_exp: float = 768.0
    pattern: str = "focus_f"


@dataclass
class ImagePlateConfig:
    x: float = 2.42
    radius: float = 0.16
    pixels: int = 64          # mesh tessellation rings (polar disc)
    sensor_res: int = 64      # pixel-grid side length; ~π/4·res² sites active in disc
    bokeh_rays: int = 4       # stencil rays fired per pixel site per call
    bokeh_stencil_frac: float = 0.25  # stencil radius = this fraction of aperture radius


@dataclass
class StageLightTubeConfig:
    # Opening point where the tube meets stage envelope.
    opening_x: float = 0.26
    opening_y: float = -0.070
    opening_z: float = 0.0
    # Tube axis points from opening toward tube back/emitter side.
    axis_x: float = 0.0
    axis_y: float = -1.0
    axis_z: float = 0.0
    depth: float = 0.48
    profile: str = "circle"  # 'circle' | 'square' | 'ngon' | 'polygon'
    bore_radius: float = 0.050
    n_sides: int = 4
    polygon_uv: Optional[np.ndarray] = None
    diffuser_enabled: bool = True
    # <= 0 means "use bore radius minus a tiny clearance".
    diffuser_radius: float = 0.0
    diffuser_thickness: float = 0.010
    diffuser_transmittance: float = 0.86
    diffuser_diffuse_frac: float = 0.82
    emitter_radius: float = 0.045
    emitter_depth_frac: float = 0.80


@dataclass
@dataclass
class PipeCSGSpec:
    center: np.ndarray
    axis_dir: np.ndarray
    t0: float
    t1: float
    kind: str = "circle"
    radius: float = 0.05
    n_sides: int = 12
    polygon_uv: Optional[np.ndarray] = None


def _axis_basis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    nw = float(np.linalg.norm(w))
    if nw < EPS:
        w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        w = w / nw
    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64) if abs(float(w[1])) < 0.8 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(w, ref)
    nu = float(np.linalg.norm(u))
    if nu < EPS:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(w, ref)
        nu = float(np.linalg.norm(u))
    u = u / max(nu, EPS)
    v = np.cross(w, u)
    v = v / max(float(np.linalg.norm(v)), EPS)
    return u, v, w


def _point_in_polygon_2d(pt: np.ndarray, poly: np.ndarray) -> bool:
    x = float(pt[0])
    y = float(pt[1])
    n = int(poly.shape[0])
    inside = False
    j = n - 1
    for i in range(n):
        xi = float(poly[i, 0])
        yi = float(poly[i, 1])
        xj = float(poly[j, 0])
        yj = float(poly[j, 1])
        cond = ((yi > y) != (yj > y))
        if cond:
            x_cross = (xj - xi) * (y - yi) / max((yj - yi), EPS) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _profile_vertices_uv(spec: PipeCSGSpec, n_profile: int) -> np.ndarray:
    kind = str(spec.kind).lower()
    if kind == "polygon" and spec.polygon_uv is not None:
        poly = np.asarray(spec.polygon_uv, dtype=np.float64)
        if poly.ndim == 2 and poly.shape[0] >= 3 and poly.shape[1] == 2:
            return np.ascontiguousarray(poly, dtype=np.float64)
    if kind == "ngon":
        n = int(max(3, spec.n_sides))
    else:
        n = int(max(8, n_profile))
    r = float(max(1.0e-6, spec.radius))
    pts = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        t = (2.0 * math.pi * i) / float(n)
        pts[i, 0] = r * math.cos(t)
        pts[i, 1] = r * math.sin(t)
    return pts


def _profile_radius_at_angle(spec: PipeCSGSpec, angle: float, n_profile: int = 64) -> float:
    kind = str(spec.kind).lower()
    if kind == "circle":
        return float(max(1.0e-6, spec.radius))
    poly = _profile_vertices_uv(spec, int(max(8, n_profile)))
    d = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float64)
    best = float("inf")
    n = int(poly.shape[0])
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(float(denom)) < EPS:
            continue
        # Solve s*d = a + u*e.
        s = (a[0] * e[1] - a[1] * e[0]) / denom
        u = (a[0] * d[1] - a[1] * d[0]) / denom
        if s > 0.0 and -1.0e-9 <= u <= 1.0 + 1.0e-9:
            best = min(best, float(s))
    if math.isfinite(best):
        return best
    return float(max(1.0e-6, spec.radius))


def _profile_inradius(spec: PipeCSGSpec, n_profile: int = 64) -> float:
    kind = str(spec.kind).lower()
    if kind == "circle":
        return float(max(1.0e-6, spec.radius))
    poly = _profile_vertices_uv(spec, int(max(8, n_profile)))
    n = int(poly.shape[0])
    best = float("inf")
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        le = float(np.linalg.norm(e))
        if le < EPS:
            continue
        best = min(best, abs(float(a[0] * e[1] - a[1] * e[0])) / le)
    return float(max(1.0e-6, best if math.isfinite(best) else spec.radius))


def _pipe_contains_point(spec: PipeCSGSpec, p_world: np.ndarray, n_profile: int = 64) -> bool:
    c = np.asarray(spec.center, dtype=np.float64).reshape(3)
    u, v, w = _axis_basis(np.asarray(spec.axis_dir, dtype=np.float64).reshape(3))
    d = np.asarray(p_world, dtype=np.float64).reshape(3) - c
    t = float(np.dot(d, w))
    ta = float(min(spec.t0, spec.t1))
    tb = float(max(spec.t0, spec.t1))
    if t < ta - 1.0e-7 or t > tb + 1.0e-7:
        return False
    uu = float(np.dot(d, u))
    vv = float(np.dot(d, v))
    kind = str(spec.kind).lower()
    if kind == "circle":
        return (uu * uu + vv * vv) <= float(spec.radius) * float(spec.radius)
    poly = _profile_vertices_uv(spec, n_profile)
    return _point_in_polygon_2d(np.array([uu, vv], dtype=np.float64), poly)


def _build_pipe_walls_csg(
    host: PipeCSGSpec,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    *,
    n_axial: int = 64,
    n_profile: int = 64,
    cutters: Optional[Sequence[PipeCSGSpec]] = None,
    csg_mode: str = "subtract",
    tri_ids: List[int] | None = None,
) -> None:
    c = np.asarray(host.center, dtype=np.float64).reshape(3)
    u, v, w = _axis_basis(np.asarray(host.axis_dir, dtype=np.float64).reshape(3))
    prof = _profile_vertices_uv(host, int(max(8, n_profile)))
    m = int(prof.shape[0])
    t_vals = np.linspace(float(host.t0), float(host.t1), int(max(1, n_axial)) + 1, dtype=np.float64)
    cut_list = list(cutters) if cutters is not None else []
    mode = str(csg_mode).lower()

    for it in range(int(max(1, n_axial))):
        ta = float(t_vals[it])
        tb = float(t_vals[it + 1])
        for ip in range(m):
            jp = (ip + 1) % m
            a0 = prof[ip, 0] * u + prof[ip, 1] * v
            a1 = prof[jp, 0] * u + prof[jp, 1] * v
            p00 = c + ta * w + a0
            p01 = c + ta * w + a1
            p10 = c + tb * w + a0
            p11 = c + tb * w + a1

            ctr = (p00 + p01 + p10 + p11) * 0.25
            n_inside = 0
            for cut in cut_list:
                if _pipe_contains_point(cut, ctr, n_profile=int(max(8, n_profile))):
                    n_inside += 1

            if mode == "subtract":
                keep = (n_inside == 0)
            elif mode == "intersect":
                keep = (n_inside > 0)
            elif mode == "xor":
                keep = (n_inside % 2 == 1)
            else:
                keep = True

            if not keep:
                continue

            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_pipe_cap(
    spec: PipeCSGSpec,
    t: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    *,
    n_profile: int = 64,
    face_toward_pos_t: bool = True,
) -> None:
    c = np.asarray(spec.center, dtype=np.float64).reshape(3)
    u, v, w = _axis_basis(np.asarray(spec.axis_dir, dtype=np.float64).reshape(3))
    prof = _profile_vertices_uv(spec, int(max(8, n_profile)))
    center = c + float(t) * w
    ring = [center + p[0] * u + p[1] * v for p in prof]
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        if face_toward_pos_t:
            _append_tri(tri_list, mat_ids, center, ring[i], ring[j], mat_idx)
        else:
            _append_tri(tri_list, mat_ids, center, ring[j], ring[i], mat_idx)


def _tube_profile_to_kind(cfg: StageLightTubeConfig) -> Tuple[str, int]:
    p = str(cfg.profile).lower().strip()
    if p == "square":
        return "ngon", 4
    if p == "ngon":
        return "ngon", int(max(3, cfg.n_sides))
    if p == "polygon":
        return "polygon", int(max(3, cfg.n_sides))
    return "circle", int(max(8, cfg.n_sides))


def _build_stage_light_tube(
    cfg: StageLightTubeConfig,
    *,
    idx_wall: int,
    idx_diffuser: int,
    idx_source: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    source_tri_ids: List[int],
) -> PipeCSGSpec:
    open_pt = np.array([float(cfg.opening_x), float(cfg.opening_y), float(cfg.opening_z)], dtype=np.float64)
    axis = np.array([float(cfg.axis_x), float(cfg.axis_y), float(cfg.axis_z)], dtype=np.float64)
    na = float(np.linalg.norm(axis))
    if na < EPS:
        axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        axis = axis / na
    depth = float(max(0.02, cfg.depth))
    kind, n_sides = _tube_profile_to_kind(cfg)
    tube_spec = PipeCSGSpec(
        center=open_pt,
        axis_dir=axis,
        t0=0.0,
        t1=depth,
        kind=kind,
        radius=float(cfg.bore_radius),
        n_sides=int(n_sides),
        polygon_uv=None if cfg.polygon_uv is None else np.asarray(cfg.polygon_uv, dtype=np.float64),
    )
    _build_pipe_walls_csg(
        tube_spec,
        tri_list,
        mat_ids,
        idx_wall,
        n_axial=48,
        n_profile=64,
        cutters=None,
        csg_mode="union",
    )
    _build_pipe_cap(
        tube_spec,
        float(depth),
        tri_list,
        mat_ids,
        idx_wall,
        n_profile=64,
        face_toward_pos_t=False,
    )
    _build_pipe_cap(
        tube_spec,
        float(depth),
        tri_list,
        mat_ids,
        idx_wall,
        n_profile=64,
        face_toward_pos_t=True,
    )

    bore_r = float(max(1.0e-4, cfg.bore_radius))
    diffuser_fit_r = float(max(1.0e-4, _profile_inradius(tube_spec, n_profile=64) - 2.0e-4))

    if bool(cfg.diffuser_enabled):
        # The diffuser belongs to the light tube, not the stage volume. Keep
        # its stage-side face just inside the tube mouth so it cannot protrude
        # into the chamber or appear as an object outside the stage boundary.
        diff_thick = float(max(1.0e-5, cfg.diffuser_thickness))
        diff_half = 0.5 * diff_thick
        diff_clearance = float(max(1.0e-6, min(1.0e-4, 0.01 * diff_thick)))
        diff_inset = float(diff_half + diff_clearance)
        diff_center = open_pt + axis * diff_inset
        diff_r_cfg = float(cfg.diffuser_radius)
        diff_r = float(np.clip(
            diff_r_cfg if diff_r_cfg > 0.0 else diffuser_fit_r,
            1.0e-4,
            diffuser_fit_r,
        ))
        _build_thin_disc_element_oriented(
            center=np.ascontiguousarray(diff_center, dtype=np.float64),
            normal=np.ascontiguousarray(axis, dtype=np.float64),
            radius=diff_r,
            thickness=diff_thick,
            tri_list=tri_list,
            mat_ids=mat_ids,
            mat_idx=idx_diffuser,
            n_theta=96,
        )

    emit_t = float(np.clip(cfg.emitter_depth_frac, 0.55, 0.97)) * depth
    emit_center = open_pt + axis * emit_t
    # Allow emitter to fill up to 85% of bore; previously 62% made the emitter
    # too small relative to the chamber and left gaps that leaked light through
    # the back of the tube when the configured emitter_radius exceeded the cap.
    emit_r = float(np.clip(cfg.emitter_radius, 1.0e-4, max(1.0e-4, 0.85 * bore_r)))

    # Mirror-lined emitter cavity: physically funnels emissive flux toward the
    # tube mouth without Python-side launch hacks.
    # Scale cavity to always be at least 1.10 * emit_r to avoid the emitter
    # surface intersecting the cavity walls.
    cavity_len = float(np.clip(0.22 * depth, 0.050, 0.150))
    cavity_t0 = float(np.clip(depth - cavity_len, 0.0, max(0.0, depth - 1.0e-4)))
    cavity_t1 = float(np.clip(depth - 1.0e-4, cavity_t0 + 1.0e-4, depth))
    cavity_r = float(max(1.10 * emit_r, 0.95 * bore_r))
    cavity_spec = PipeCSGSpec(
        center=open_pt,
        axis_dir=axis,
        t0=cavity_t0,
        t1=cavity_t1,
        kind="ngon",
        radius=cavity_r,
        n_sides=4,
    )
    _build_pipe_walls_csg(
        cavity_spec,
        tri_list,
        mat_ids,
        idx_wall,
        n_axial=16,
        n_profile=16,
        cutters=None,
        csg_mode="union",
    )

    _build_emissive_sphere(
        center=np.ascontiguousarray(emit_center, dtype=np.float64),
        radius=emit_r,
        tri_list=tri_list,
        mat_ids=mat_ids,
        mat_idx=idx_source,
        n_theta=18,
        n_phi=10,
        tri_ids=source_tri_ids,
    )
    return tube_spec


# ─────────────────────────────────────────────────────────────────────────────
# Depth-of-Field (DOF) and Optical Math — First Principles
# ─────────────────────────────────────────────────────────────────────────────

def compute_dof_hyperfocal(focal_m: float, f_number: float, coc_m: float = 30e-6) -> float:
    """Hyperfocal distance: closest focus where infinity is acceptably sharp.

    H = f² / (N × CoC) + f
    where f = focal length, N = f-number, CoC = circle of confusion (m).

    For deep DOF at macro scales, hyperfocal is a useful reference.
    Example: 50mm f/2.8 with 30µm CoC → H ≈ 0.92 m (everything sharp from ~0.46 m ∞).
    """
    if f_number <= 0 or focal_m <= 0:
        return float('inf')
    return (focal_m * focal_m) / (f_number * coc_m) + focal_m


def compute_dof_near_far(focal_m: float, f_number: float, focus_dist_m: float,
                          coc_m: float = 30e-6) -> tuple:
    """Near and far DOF limits: closest and farthest distances that appear sharp.

    DOF_near = (f × s × N) / (f² + N × CoC × (s - f))
    DOF_far  = (f × s × N) / (f² - N × CoC × (s - f))

    s = focus distance, f = focal length, N = f-number, CoC = circle of confusion.

    Returns (near_m, far_m). If far_m > 100 m, it's effectively infinity.
    Used to verify aperture size is sufficient for desired scene depth.
    """
    if focal_m <= 0 or f_number <= 0 or focus_dist_m <= focal_m:
        return 0.0, float('inf')
    numer = focal_m * focus_dist_m * f_number
    denom_sq = focal_m * focal_m
    denom_coc = f_number * coc_m * (focus_dist_m - focal_m)
    near = numer / (denom_sq + denom_coc) if (denom_sq + denom_coc) > 0 else 0.0
    far = numer / max(denom_sq - denom_coc, 1e-9) if (denom_sq - denom_coc) > 0 else float('inf')
    return near, far


def compute_required_aperture_for_dof(focal_m: float, focus_dist_m: float,
                                       desired_dof_m: float,
                                       coc_m: float = 30e-6) -> float:
    """Reverse-compute f-number needed to achieve a given DOF.

    Given focal length, focus distance, and desired depth of field,
    compute the required f-number (and thus aperture radius).

    Approximate formula: N ≈ (2 × f × (s - f) × CoC) / DOF
    Returns f-number; divide focal_m by (2 × result) to get aperture_radius.

    Example: 50mm lens at 0.3 m focus, needing 0.05 m DOF → ~f/8.5 → aperture ≈ 3 mm.
    """
    if focal_m <= 0 or focus_dist_m <= focal_m or desired_dof_m <= 0:
        return 1.0
    return (2.0 * focal_m * (focus_dist_m - focal_m) * coc_m) / desired_dof_m


@dataclass
class IrisApertureConfig:
    """Parametric iris diaphragm — the camera designer's ApertureStop model bridged
    into the ray tracer as physical blade geometry.

    Coordinate system matches thick_lens_focus_lab: optical axis = X, transverse = Y/Z.

    n_blades == 0  → circular annular aperture stop (simple annulus).
    n_blades >= 3  → N-blade iris; each blade covers 1.5× its angular slot so
                     adjacent blades overlap and close the annular region fully,
                     matching the ApertureStop polygon model from the camera designer.

    Triangles are registered as opaque (MAT_FLAG_APERTURE_STOP) BLOCKER geometry
    so the pixel-cone path tests real blade shape, not a circular fallback disk.

    Use ``iris_from_camera_preset(preset, x_pos)`` to populate this directly from
    a camera_designer CameraPreset.aperture_stop, closing the design loop.
    """
    enabled: bool = True
    x_pos: float = 1.12          # optical-axis position (metres, inside lens stack)
    r_inner: float = 0.018       # clear aperture radius (metres)
    r_outer: float = 0.028       # blade outer radius — opaque beyond this to bore wall
    n_blades: int = 6            # 0 = circle; >=3 = regular N-gon blade set
    rotation_deg: float = 0.0    # first-blade edge angle (degrees)


@dataclass
class SceneConfig:
    x_min: float = 0.0
    x_max: float = 2.6
    source_x: float = 0.08
    source_radius: float = 0.050
    tube_x0: float = 0.12
    tube_x1: float = 1.80
    tube_radius: float = 0.070
    baffle0_x: float = 0.12  # Right after source, collimating aperture
    baffle0_aperture: float = 0.1
    baffle1_x: float = 1.15  # After lens back at 1.13, before screen
    baffle1_aperture: float = 0.1
    baffle2_x: float = 2.35  # Just before screen at 2.42
    baffle2_aperture: float = 0.1
    screen_x: float = 2.42
    screen_radius: float = 0.16
    view_radius: float = 0.24
    enable_debug_plate: bool = False
    debug_plate_x: float = 0.50
    debug_plate_radius: float = 0.20
    object_plane: ObjectPlaneConfig = field(default_factory=ObjectPlaneConfig)
    image_plate: ImagePlateConfig = field(default_factory=ImagePlateConfig)
    side_room_y: float = -0.090
    side_room_aperture: float = 0.035
    side_room_bore_radius: float = 0.064
    side_room_source_radius: float = 0.030
    side_room_wall_outer: float = 0.220
    side_room_depth: float = 0.56
    side_room_source_offset: float = 0.22
    side_room_source_emission: float = 12.0
    side_room_diffuser_enabled: bool = True
    side_room_diffuser_radius: float = 0.060
    side_room_diffuser_thickness: float = 0.010
    side_room_diffuser_transmittance: float = 0.86
    side_room_diffuser_diffuse_frac: float = 0.82
    side_room_diffuser_tilt_x_deg: float = 16.0
    side_room_diffuser_tilt_z_deg: float = -13.0
    side_room_source_x: float = 0.26
    side_room_source_z: float = 0.00
    # Large emissive probe object placed in the stage volume to verify that the
    # lens train is doing anything intelligible independent of the side tube.
    stage_probe_emitter_enabled: bool = False
    stage_probe_x: float = 0.62
    stage_probe_y: float = 0.00
    stage_probe_z: float = 0.00
    stage_probe_radius: float = 0.045
    # Stage wall interior calibration material (18% event exposure grey).
    stage_grey_reflectance: float = 0.18
    # Macro-configurable stage lighting tubes (additional to legacy side-room).
    stage_light_tubes: List[StageLightTubeConfig] = field(default_factory=list)
    # Parametric iris diaphragm: None = use simple circular baffle0 as aperture stop.
    # Set to an IrisApertureConfig (or use iris_from_camera_preset()) to register
    # blade-polygon geometry as the BLOCKER aperture stop for BDPT pixel-cone tests.
    iris_aperture: Optional[IrisApertureConfig] = None
    # Exit pupil / field stop aperture between last lens and sensor. When enabled
    # this limits light transmission post-optics, letting you tune spectral content
    # and transmission efficiency independently of the entrance aperture.
    exit_pupil_x: float = 1.85
    exit_pupil_radius: float = 0.008   # 8 mm radius
    exit_pupil_thickness: float = 0.0003  # 0.3 mm — blade aperture
    # Optional supplementary macro lens (close focus helper): inserted at the front
    # to extend working distance and achieve extreme magnification.
    # Leave None to use standard lens stack; set to a LensConfig to add macro element.
    macro_lens: Optional[LensConfig] = None
    disable_optics: bool = False
    lens: LensConfig = field(
        default_factory=lambda: LensConfig(
            center_x=1.08,
            thickness=0.050,
            aperture_radius=0.040,
            radius_front=0.10,
            radius_back=0.10,
            ior=1.52,
        )
    )
    lens_stack: List[LensConfig] = field(default_factory=lambda: [
        LensConfig(center_x=0.98, thickness=0.040, aperture_radius=0.036, radius_front=0.090, radius_back=0.110, ior=1.52),
        LensConfig(center_x=1.08, thickness=0.030, aperture_radius=0.030, radius_front=0.140, radius_back=0.140, ior=1.62),
        LensConfig(center_x=1.19, thickness=0.040, aperture_radius=0.034, radius_front=0.100, radius_back=0.090, ior=1.52),
        LensConfig(center_x=1.34, thickness=0.045, aperture_radius=0.038, radius_front=0.120, radius_back=0.120, ior=1.57),
    ])


def _scene_lenses(scene: SceneConfig) -> List[LensConfig]:
    lenses = list(getattr(scene, "lens_stack", []) or [])
    return lenses if lenses else [scene.lens]


def _lens_is_valid(lens: LensConfig, min_edge_thickness: float = 0.004) -> bool:
    if lens.thickness <= 0.0 or lens.aperture_radius <= 0.0:
        return False
    if abs(lens.radius_front) <= lens.aperture_radius or abs(lens.radius_back) <= lens.aperture_radius:
        return False
    r = np.asarray([0.0, lens.aperture_radius], dtype=np.float64)
    return bool(np.all(_lens_back_x(lens, r) - _lens_front_x(lens, r) >= min_edge_thickness))


@dataclass
class SpectralRibbonSource:
    x: float
    y_min: float
    y_max: float
    z: float = 0.0
    n_emitters: int = 64
    directivity_exp: float = 512.0

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> "SpectralRibbonSource":
        r = float(scene.source_radius)
        return cls(
            x=float(scene.source_x) + 1.0e-4,
            y_min=-0.92 * r,
            y_max=0.92 * r,
            z=0.0,
            n_emitters=64,
            directivity_exp=512.0,
        )

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = max(1, int(self.n_emitters))
        ys = np.linspace(self.y_min, self.y_max, n, dtype=np.float64)
        src_pos = np.column_stack([
            np.full((n,), self.x, dtype=np.float64),
            ys,
            np.full((n,), self.z, dtype=np.float64),
        ])
        src_dir = np.zeros((n, 3), dtype=np.float64)
        src_dir[:, 0] = 1.0
        src_directivity = np.full((n,), float(self.directivity_exp), dtype=np.float64)
        return (
            np.ascontiguousarray(src_pos, dtype=np.float64),
            np.ascontiguousarray(src_dir, dtype=np.float64),
            np.ascontiguousarray(src_directivity, dtype=np.float64),
        )


def _object_plane_mask(cfg: ObjectPlaneConfig) -> np.ndarray:
    n = int(max(3, cfg.pixels))
    yy, zz = np.mgrid[0:n, 0:n]
    u = yy / max(1, n - 1)
    v = zz / max(1, n - 1)
    if cfg.pattern == "cross":
        mask = (np.abs(u - 0.5) < 0.08) | (np.abs(v - 0.5) < 0.08)
    else:
        # Blocky "F" target: asymmetric, so focus and inversion are visible.
        mask = (
            (u < 0.18)
            | ((v < 0.18) & (u < 0.90))
            | ((np.abs(v - 0.48) < 0.08) & (u < 0.70))
        )
    rr = (2.0 * u - 1.0) ** 2 + (2.0 * v - 1.0) ** 2
    return np.asarray(mask & (rr <= 1.0), dtype=bool)


@dataclass
class LitObjectPlaneSource:
    cfg: ObjectPlaneConfig

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> "LitObjectPlaneSource":
        return cls(scene.object_plane)

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = int(max(3, cfg.pixels))
        mask = _object_plane_mask(cfg)
        coords = np.linspace(-float(cfg.radius), float(cfg.radius), n, dtype=np.float64)
        yy, zz = np.meshgrid(coords, coords, indexing="ij")
        if np.any(mask):
            yv = yy[mask]
            zv = zz[mask]
            xv = np.full((int(yv.size),), float(cfg.x) + 1.0e-4, dtype=np.float64)
            src_pos = np.column_stack([xv, yv, zv]).astype(np.float64, copy=False)
        else:
            src_pos = np.asarray([(float(cfg.x) + 1.0e-4, 0.0, 0.0)], dtype=np.float64)
        src_dir = np.zeros_like(src_pos, dtype=np.float64)
        src_dir[:, 0] = 1.0
        src_directivity = np.full((src_pos.shape[0],), float(cfg.directivity_exp), dtype=np.float64)
        return (
            np.ascontiguousarray(src_pos, dtype=np.float64),
            np.ascontiguousarray(src_dir, dtype=np.float64),
            np.ascontiguousarray(src_directivity, dtype=np.float64),
        )


def _normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    cr = np.cross(b - a, c - a)
    n = float(np.linalg.norm(cr))
    if n < EPS:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return cr / n


def _append_tri(
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    mat_idx: int,
) -> None:
    tri_list.append(np.array([a, b, c], dtype=np.float64))
    mat_ids.append(int(mat_idx))


def _build_disc_source(
    scene: SceneConfig,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    source_mat: int,
    n_theta: int = 48,
) -> None:
    x = float(scene.source_x)
    r = float(scene.source_radius)
    center = np.array([x, 0.0, 0.0], dtype=np.float64)

    ring = []
    for i in range(n_theta):
        t = (2.0 * math.pi * i) / float(n_theta)
        ring.append(np.array([x, r * math.cos(t), r * math.sin(t)], dtype=np.float64))

    for i in range(n_theta):
        p1 = ring[i]
        p2 = ring[(i + 1) % n_theta]
        _append_tri(tri_list, mat_ids, center, p1, p2, source_mat)


def _build_disc_cap(
    x: float,
    radius: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    cap_mat: int,
    n_theta: int = 48,
    tri_ids: List[int] | None = None,
) -> None:
    """Flat disc at *x* with normal pointing +x (reflects back toward the lens)."""
    center = np.array([x, 0.0, 0.0], dtype=np.float64)
    ring = [
        np.array([x, radius * math.cos(2.0 * math.pi * i / n_theta),
                     radius * math.sin(2.0 * math.pi * i / n_theta)], dtype=np.float64)
        for i in range(n_theta)
    ]
    for i in range(n_theta):
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, center, ring[i], ring[(i + 1) % n_theta], cap_mat)


def _build_cylinder_walls(
    x0: float,
    x1: float,
    radius: float,
    n_theta: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    tri_ids: List[int] | None = None,
) -> None:
    for i in range(n_theta):
        t0 = (2.0 * math.pi * i) / float(n_theta)
        t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)

        y0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        y1, z1 = radius * math.cos(t1), radius * math.sin(t1)

        p00 = np.array([x0, y0, z0], dtype=np.float64)
        p01 = np.array([x0, y1, z1], dtype=np.float64)
        p10 = np.array([x1, y0, z0], dtype=np.float64)
        p11 = np.array([x1, y1, z1], dtype=np.float64)

        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_cylinder_walls_with_hole(
    x0: float, x1: float, radius: float, n_theta: int,
    tri_list: List[np.ndarray], mat_ids: List[int], mat_idx: int,
    hole_x_center: float, hole_x_half: float,
    hole_theta_center: float, hole_theta_half: float,
    n_x_sub: int = 32,
    tri_ids: List[int] | None = None,
) -> None:
    """Cylinder wall along X with a circular cutout on its surface.

    Subdivides the x span so that panels whose centre falls inside
    (hole_x_center ± hole_x_half) × (hole_theta_center ± hole_theta_half)
    are omitted — cutting the hole-saw opening into the pipe wall.
    """
    xs = np.linspace(x0, x1, n_x_sub + 1)
    for ix in range(n_x_sub):
        xa, xb = float(xs[ix]), float(xs[ix + 1])
        xm = 0.5 * (xa + xb)
        in_x = abs(xm - hole_x_center) < hole_x_half
        for i in range(n_theta):
            t0 = (2.0 * math.pi * i) / float(n_theta)
            t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)
            if in_x:
                tm = 0.5 * (t0 + t1)
                dt = (tm - hole_theta_center + math.pi) % (2.0 * math.pi) - math.pi
                if abs(dt) < hole_theta_half:
                    continue  # inside the hole — skip this panel
            y0, z0 = radius * math.cos(t0), radius * math.sin(t0)
            y1, z1 = radius * math.cos(t1), radius * math.sin(t1)
            p00 = np.array([xa, y0, z0], dtype=np.float64)
            p01 = np.array([xa, y1, z1], dtype=np.float64)
            p10 = np.array([xb, y0, z0], dtype=np.float64)
            p11 = np.array([xb, y1, z1], dtype=np.float64)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_baffle_annulus(
    x: float,
    r_inner: float,
    r_outer: float,
    n_theta: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    tri_ids: List[int] | None = None,
    thickness: float = 0.008,
) -> None:
    """Flat annulus baffle with an inner bore wall of `thickness` depth.

    The bore cylinder (from *x* to *x+thickness* at *r_inner*) catches grazing
    rays that would otherwise slip past the infinitely-thin inner edge.
    """
    # Front face (existing geometry, unchanged)
    for i in range(n_theta):
        t0 = (2.0 * math.pi * i) / float(n_theta)
        t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)

        i0 = np.array([x, r_inner * math.cos(t0), r_inner * math.sin(t0)], dtype=np.float64)
        i1 = np.array([x, r_inner * math.cos(t1), r_inner * math.sin(t1)], dtype=np.float64)
        o0 = np.array([x, r_outer * math.cos(t0), r_outer * math.sin(t0)], dtype=np.float64)
        o1 = np.array([x, r_outer * math.cos(t1), r_outer * math.sin(t1)], dtype=np.float64)

        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, i0, o0, o1, mat_idx)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, i0, o1, i1, mat_idx)

    # Back face at x+thickness (catches rays approaching from behind)
    if thickness > 0.0:
        xb = x + thickness
        for i in range(n_theta):
            t0 = (2.0 * math.pi * i) / float(n_theta)
            t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)

            i0 = np.array([xb, r_inner * math.cos(t0), r_inner * math.sin(t0)], dtype=np.float64)
            i1 = np.array([xb, r_inner * math.cos(t1), r_inner * math.sin(t1)], dtype=np.float64)
            o0 = np.array([xb, r_outer * math.cos(t0), r_outer * math.sin(t0)], dtype=np.float64)
            o1 = np.array([xb, r_outer * math.cos(t1), r_outer * math.sin(t1)], dtype=np.float64)

            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, i0, o1, o0, mat_idx)  # reversed winding → +x normal
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, i0, i1, o1, mat_idx)

        # Inner bore cylinder: catches grazing rays that slip through the aperture hole
        # traveling at oblique angles to the baffle plane.
        for i in range(n_theta):
            t0 = (2.0 * math.pi * i) / float(n_theta)
            t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)

            y0, z0 = r_inner * math.cos(t0), r_inner * math.sin(t0)
            y1, z1 = r_inner * math.cos(t1), r_inner * math.sin(t1)

            p00 = np.array([x,  y0, z0], dtype=np.float64)
            p01 = np.array([x,  y1, z1], dtype=np.float64)
            p10 = np.array([xb, y0, z0], dtype=np.float64)
            p11 = np.array([xb, y1, z1], dtype=np.float64)

            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)  # inward-facing normal
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_sensor_aperture_frustum(
    x_aperture: float,
    r_aperture: float,
    x_sensor: float,
    r_sensor: float,
    n_theta: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    tri_ids: List[int] | None = None,
) -> None:
    """Opaque truncated-cone wall connecting aperture stop extents to sensor extents.

    Rays that pass through the aperture opening but travel outside the cone defined
    by (x_aperture, r_aperture) -> (x_sensor, r_sensor) will hit these walls and be
    absorbed, physically enforcing the aperture acceptance solid angle in mesh space.
    The cone is double-sided (inner and outer rings) so both forward and reverse rays
    participate in collisions on every bounce.
    """
    xa = float(x_aperture)
    ra = float(r_aperture)
    xs = float(x_sensor)
    rs = float(r_sensor)
    n = int(max(8, n_theta))

    for i in range(n):
        t0 = (2.0 * math.pi * i) / float(n)
        t1 = (2.0 * math.pi * (i + 1)) / float(n)
        # Aperture-end ring
        a0 = np.array([xa, ra * math.cos(t0), ra * math.sin(t0)], dtype=np.float64)
        a1 = np.array([xa, ra * math.cos(t1), ra * math.sin(t1)], dtype=np.float64)
        # Sensor-end ring
        s0 = np.array([xs, rs * math.cos(t0), rs * math.sin(t0)], dtype=np.float64)
        s1 = np.array([xs, rs * math.cos(t1), rs * math.sin(t1)], dtype=np.float64)

        # Inward-facing face (seen by rays travelling from scene toward sensor)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a0, s0, s1, mat_idx)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a0, s1, a1, mat_idx)

        # Outward-facing face (seen by reverse rays escaping from sensor side)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a0, a1, s1, mat_idx)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a0, s1, s0, mat_idx)


def _build_iris_baffle(
    cfg: "IrisApertureConfig",
    tube_radius: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    idx_black: int,
    tri_ids_out: List[int],
) -> None:
    """Tessellate a parametric iris diaphragm as opaque blocker triangles.

    The blade geometry is ported from camera_designer/scene_builder.py and adapted
    for the X-axis optical coordinate system of thick_lens_focus_lab.py (optical
    axis = X, transverse = Y/Z).

    Blade sectors fill r_inner → r_outer using 1.5× overlapping angular spans so
    adjacent blades close the annular opening completely.  A solid outer annulus
    r_outer → tube_radius seals the bore wall.  All triangles are registered into
    tri_ids_out so the pixel-cone path can test real blade occlusion geometry.
    """
    x      = float(cfg.x_pos)
    r_in   = float(cfg.r_inner)
    r_out  = float(cfg.r_outer)
    tube_r = float(max(tube_radius, r_out + 1e-4))
    n_bl   = int(cfg.n_blades)
    a_rot  = float(cfg.rotation_deg) * math.pi / 180.0

    def _pt(r: float, th: float) -> np.ndarray:
        return np.array([x, r * math.cos(th), r * math.sin(th)], np.float64)

    def _add_tri(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        tri_ids_out.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a, b, c, idx_black)

    # ── Blade sectors (r_inner → r_outer) ────────────────────────────────
    if n_bl >= 3:
        sector           = 2.0 * math.pi / n_bl
        blade_half_angle = sector * 0.75   # 1.5× slot → adjacent blades overlap
        n_tang = 80
        n_rad  = 24
        r_start = r_in if r_in > 1e-9 else r_out * 0.05
        for bi in range(n_bl):
            pivot  = a_rot + bi * sector
            th_min = pivot - blade_half_angle
            th_max = pivot + blade_half_angle
            thetas = np.linspace(th_min, th_max, n_tang + 1)
            radii  = np.linspace(r_start, r_out, n_rad + 1)
            pts    = np.zeros((n_rad + 1, n_tang + 1, 3), np.float64)
            for ri, r in enumerate(radii):
                for ti, th in enumerate(thetas):
                    pts[ri, ti] = _pt(r, th)
            for ri in range(n_rad):
                for ti in range(n_tang):
                    p00 = pts[ri,     ti    ]
                    p10 = pts[ri + 1, ti    ]
                    p11 = pts[ri + 1, ti + 1]
                    p01 = pts[ri,     ti + 1]
                    _add_tri(p00, p10, p11)
                    _add_tri(p00, p11, p01)
    else:
        # Circular mode: opaque central disc from 0 → r_inner (plug the hole)
        if r_in > 1e-9:
            n_c    = 256
            angles = np.linspace(0.0, 2.0 * math.pi, n_c, endpoint=False)
            ctr    = _pt(0.0, 0.0)
            for i in range(n_c):
                _add_tri(ctr, _pt(r_in, angles[i]), _pt(r_in, angles[(i + 1) % n_c]))

    # ── Outer annulus: r_outer → tube_radius ─────────────────────────────
    n_rings   = 12
    n_sectors = 256
    for ri in range(n_rings):
        r0 = r_out + (tube_r - r_out) *  ri      / n_rings
        r1 = r_out + (tube_r - r_out) * (ri + 1) / n_rings
        angles = np.linspace(0.0, 2.0 * math.pi, n_sectors, endpoint=False)
        for i in range(n_sectors):
            a0 = angles[i]
            a1 = angles[(i + 1) % n_sectors]
            _add_tri(_pt(r0, a0), _pt(r1, a0), _pt(r1, a1))
            _add_tri(_pt(r0, a0), _pt(r1, a1), _pt(r0, a1))

    print(
        "[iris-baffle]",
        f"x={x:.4f}",
        f"r_inner={r_in*1e3:.2f}mm",
        f"r_outer={r_out*1e3:.2f}mm",
        f"n_blades={n_bl}",
        f"rot={cfg.rotation_deg:.1f}deg",
        f"tris={len(tri_ids_out)}",
        flush=True,
    )


def iris_from_camera_preset(preset, x_pos: float) -> "IrisApertureConfig":
    """Create an IrisApertureConfig from a camera_designer CameraPreset.

    Bridges the camera designer's ApertureStop parametric model directly into
    the ray tracer's physical iris baffle geometry, closing the design loop
    between the camera designer and the BDPT renderer.

    The ``ApertureStop.aperture_rot`` is in radians; it is converted to degrees
    for ``IrisApertureConfig.rotation_deg``.

    Example::

        from camera_designer.camera_preset import simple_doublet_preset
        iris_cfg = iris_from_camera_preset(simple_doublet_preset(), x_pos=1.12)
        scene = SceneConfig(iris_aperture=iris_cfg, ...)
    """
    ap = preset.aperture_stop
    return IrisApertureConfig(
        enabled=True,
        x_pos=float(x_pos),
        r_inner=float(ap.r_inner),
        r_outer=float(ap.r_outer),
        n_blades=int(ap.n_blades),
        rotation_deg=float(ap.aperture_rot) * 180.0 / math.pi,
    )


def _build_screen(
    scene: SceneConfig,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    tri_ids: List[int] | None = None,
) -> None:
    """Circular sensor disc tessellated as a polar-ring fan.

    Uses concentric rings so the surface is a true disc rather than a square
    grid clipped to a radius.  Triangle count is O(n_rings * n_theta).
    """
    plate  = scene.image_plate
    x      = float(plate.x)
    r      = float(plate.radius)
    n_rings = int(max(2, plate.pixels))
    # More azimuthal divisions than radial so each panel is roughly square.
    n_theta = int(max(6, n_rings * 3))

    center = np.array([x, 0.0, 0.0], dtype=np.float64)

    # Inner fan: center → first ring
    r_inner = r / n_rings
    for i in range(n_theta):
        t0 = (2.0 * math.pi * i)       / float(n_theta)
        t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)
        p1 = np.array([x, r_inner * math.cos(t0), r_inner * math.sin(t0)], dtype=np.float64)
        p2 = np.array([x, r_inner * math.cos(t1), r_inner * math.sin(t1)], dtype=np.float64)
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, center, p1, p2, mat_idx)

    # Annular strips: ring k → ring k+1
    for ring in range(1, n_rings):
        ra = ring       * r / n_rings
        rb = (ring + 1) * r / n_rings
        for i in range(n_theta):
            t0 = (2.0 * math.pi * i)       / float(n_theta)
            t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)
            a0 = np.array([x, ra * math.cos(t0), ra * math.sin(t0)], dtype=np.float64)
            a1 = np.array([x, ra * math.cos(t1), ra * math.sin(t1)], dtype=np.float64)
            b0 = np.array([x, rb * math.cos(t0), rb * math.sin(t0)], dtype=np.float64)
            b1 = np.array([x, rb * math.cos(t1), rb * math.sin(t1)], dtype=np.float64)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, a0, b0, b1, mat_idx)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, a0, b1, a1, mat_idx)


def _build_side_disc_source(
    x_center: float,
    z_center: float,
    y: float,
    radius: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    source_mat: int,
    n_theta: int = 48,
    tri_ids: List[int] | None = None,
) -> None:
    center = np.array([x_center, y, z_center], dtype=np.float64)
    ring = []
    for i in range(n_theta):
        t = (2.0 * math.pi * i) / float(n_theta)
        ring.append(np.array([
            x_center + radius * math.cos(t),
            y,
            z_center + radius * math.sin(t),
        ], dtype=np.float64))
    for i in range(n_theta):
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        # Winding emits toward +y into the pinhole wall.
        _append_tri(tri_list, mat_ids, center, ring[(i + 1) % n_theta], ring[i], source_mat)


def _build_oriented_disc_source(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    source_mat: int,
    n_theta: int = 96,
    tri_ids: List[int] | None = None,
) -> None:
    """Build a one-sided emissive disc whose emitted face points along `normal`."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < EPS:
        n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        n = n / nn

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(n[0])) < 0.8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(n, ref)
    un = float(np.linalg.norm(u))
    if un < EPS:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(n, ref)
        un = float(np.linalg.norm(u))
    u = u / max(un, EPS)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), EPS)

    ring: List[np.ndarray] = []
    rr = float(max(1.0e-5, radius))
    n_seg = int(max(8, n_theta))
    for i in range(n_seg):
        t = (2.0 * math.pi * i) / float(n_seg)
        ring.append(np.ascontiguousarray(c + rr * (math.cos(t) * u + math.sin(t) * v), dtype=np.float64))

    for i in range(n_seg):
        j = (i + 1) % n_seg
        a = c
        b = ring[i]
        d = ring[j]
        cr = np.cross(b - a, d - a)
        if float(np.dot(cr, n)) < 0.0:
            b, d = d, b
        if tri_ids is not None:
            tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, a, b, d, source_mat)


def _build_side_wall_pinhole(
    y: float,
    r_inner: float,
    r_outer: float,
    x_center: float,
    z_center: float,
    n_theta: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
) -> None:
    for i in range(n_theta):
        t0 = (2.0 * math.pi * i) / float(n_theta)
        t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)
        i0 = np.array([x_center + r_inner * math.cos(t0), y, z_center + r_inner * math.sin(t0)], dtype=np.float64)
        i1 = np.array([x_center + r_inner * math.cos(t1), y, z_center + r_inner * math.sin(t1)], dtype=np.float64)
        o0 = np.array([x_center + r_outer * math.cos(t0), y, z_center + r_outer * math.sin(t0)], dtype=np.float64)
        o1 = np.array([x_center + r_outer * math.cos(t1), y, z_center + r_outer * math.sin(t1)], dtype=np.float64)
        _append_tri(tri_list, mat_ids, i0, o1, o0, mat_idx)
        _append_tri(tri_list, mat_ids, i0, i1, o1, mat_idx)


def _build_cylinder_walls_y(
    y0: float,
    y1: float,
    radius: float,
    x_center: float,
    z_center: float,
    n_theta: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
) -> None:
    for i in range(n_theta):
        t0 = (2.0 * math.pi * i) / float(n_theta)
        t1 = (2.0 * math.pi * (i + 1)) / float(n_theta)

        x0, z0 = x_center + radius * math.cos(t0), z_center + radius * math.sin(t0)
        x1, z1 = x_center + radius * math.cos(t1), z_center + radius * math.sin(t1)

        p00 = np.array([x0, y0, z0], dtype=np.float64)
        p01 = np.array([x1, y0, z1], dtype=np.float64)
        p10 = np.array([x0, y1, z0], dtype=np.float64)
        p11 = np.array([x1, y1, z1], dtype=np.float64)

        _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)
        _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_disc_cap_y(
    y: float,
    radius: float,
    x_center: float,
    z_center: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    cap_mat: int,
    n_theta: int = 48,
    face_toward_pos_y: bool = True,
) -> None:
    center = np.array([x_center, y, z_center], dtype=np.float64)
    ring = []
    for i in range(n_theta):
        t = (2.0 * math.pi * i) / float(n_theta)
        ring.append(np.array([x_center + radius * math.cos(t), y, z_center + radius * math.sin(t)], dtype=np.float64))

    for i in range(n_theta):
        p0 = ring[i]
        p1 = ring[(i + 1) % n_theta]
        if face_toward_pos_y:
            _append_tri(tri_list, mat_ids, center, p1, p0, cap_mat)
        else:
            _append_tri(tri_list, mat_ids, center, p0, p1, cap_mat)


def _build_decorative_mesh(
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    center: np.ndarray,
    radius: float = 0.020,
    n_u: int = 28,
    n_v: int = 20,
    tri_ids: List[int] | None = None,
) -> None:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    pts = np.zeros((n_u + 1, n_v + 1, 3), dtype=np.float64)
    for iu in range(n_u + 1):
        u = (2.0 * math.pi * iu) / float(max(1, n_u))
        for iv in range(n_v + 1):
            v = (math.pi * iv) / float(max(1, n_v))
            ripple = 1.0 + 0.16 * math.sin(3.0 * u + 2.0 * v)
            r = float(radius) * ripple
            x = cx + r * math.cos(v)
            y = cy + r * math.sin(v) * math.cos(u)
            z = cz + r * math.sin(v) * math.sin(u)
            pts[iu, iv] = np.array([x, y, z], dtype=np.float64)

    for iu in range(n_u):
        for iv in range(n_v):
            p00 = pts[iu, iv]
            p01 = pts[iu, iv + 1]
            p10 = pts[iu + 1, iv]
            p11 = pts[iu + 1, iv + 1]
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p10, p11, mat_idx)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, p00, p11, p01, mat_idx)


def _build_emissive_sphere(
    center: np.ndarray,
    radius: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    n_theta: int = 48,
    n_phi: int = 24,
    tri_ids: List[int] | None = None,
) -> None:
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    c = np.array([cx, cy, cz], dtype=np.float64)

    def _append_outward(a: np.ndarray, b: np.ndarray, c_pt: np.ndarray) -> None:
        n = np.cross(b - a, c_pt - a)
        tri_ctr = (a + b + c_pt) / 3.0
        # Ensure outward winding relative to sphere center.
        if float(np.dot(n, tri_ctr - c)) < 0.0:
            _append_tri(tri_list, mat_ids, a, c_pt, b, mat_idx)
        else:
            _append_tri(tri_list, mat_ids, a, b, c_pt, mat_idx)

    pts = np.zeros((n_theta + 1, n_phi + 1, 3), dtype=np.float64)
    for it in range(n_theta + 1):
        t = (2.0 * math.pi * it) / float(max(1, n_theta))
        ct = math.cos(t)
        st = math.sin(t)
        for ip in range(n_phi + 1):
            p = (math.pi * ip) / float(max(1, n_phi))
            sp = math.sin(p)
            cp = math.cos(p)
            pts[it, ip] = np.array([
                cx + radius * sp * ct,
                cy + radius * cp,
                cz + radius * sp * st,
            ], dtype=np.float64)

    for it in range(n_theta):
        for ip in range(n_phi):
            p00 = pts[it, ip]
            p01 = pts[it, ip + 1]
            p10 = pts[it + 1, ip]
            p11 = pts[it + 1, ip + 1]
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_outward(p00, p10, p11)
            if tri_ids is not None:
                tri_ids.append(len(tri_list))
            _append_outward(p00, p11, p01)


def _build_thin_disc_element_y(
    y_center: float,
    radius: float,
    thickness: float,
    x_center: float,
    z_center: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    n_theta: int = 72,
) -> None:
    t = max(1.0e-5, float(thickness))
    y0 = float(y_center) - 0.5 * t
    y1 = float(y_center) + 0.5 * t
    _build_disc_cap_y(
        y=y0,
        radius=float(radius),
        x_center=float(x_center),
        z_center=float(z_center),
        tri_list=tri_list,
        mat_ids=mat_ids,
        cap_mat=mat_idx,
        n_theta=n_theta,
        face_toward_pos_y=False,
    )
    _build_disc_cap_y(
        y=y1,
        radius=float(radius),
        x_center=float(x_center),
        z_center=float(z_center),
        tri_list=tri_list,
        mat_ids=mat_ids,
        cap_mat=mat_idx,
        n_theta=n_theta,
        face_toward_pos_y=True,
    )
    _build_cylinder_walls_y(
        y0,
        y1,
        float(radius),
        float(x_center),
        float(z_center),
        n_theta,
        tri_list,
        mat_ids,
        mat_idx,
    )


def _build_thin_disc_element_oriented(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    thickness: float,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    n_theta: int = 72,
) -> None:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < EPS:
        n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        n = n / nn

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(n[0])) < 0.8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(n, ref)
    un = float(np.linalg.norm(u))
    if un < EPS:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(n, ref)
        un = float(np.linalg.norm(u))
    u = u / max(un, EPS)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), EPS)

    h = 0.5 * max(1.0e-5, float(thickness))
    c0 = c - h * n
    c1 = c + h * n

    ring0: List[np.ndarray] = []
    ring1: List[np.ndarray] = []
    for i in range(n_theta):
        t = (2.0 * math.pi * i) / float(max(1, n_theta))
        radial = float(radius) * (math.cos(t) * u + math.sin(t) * v)
        ring0.append(np.ascontiguousarray(c0 + radial, dtype=np.float64))
        ring1.append(np.ascontiguousarray(c1 + radial, dtype=np.float64))

    for i in range(n_theta):
        j = (i + 1) % n_theta
        # c0 = center - h*n: stage-side face, normal must point away from disc toward -n.
        # Reversed winding flips the normal from +n to -n so front-face is visible from outside.
        _append_tri(tri_list, mat_ids, c0, ring0[j], ring0[i], mat_idx)
        # c1 = center + h*n: tube-interior-side face, normal must point away from disc toward +n.
        # Reversed winding flips the normal from -n to +n so front-face is visible from tube interior.
        _append_tri(tri_list, mat_ids, c1, ring1[i], ring1[j], mat_idx)
        # Side walls: outward-pointing radial normals are correct as-is.
        _append_tri(tri_list, mat_ids, ring0[i], ring1[i], ring1[j], mat_idx)
        _append_tri(tri_list, mat_ids, ring0[i], ring1[j], ring0[j], mat_idx)


def _lens_front_x(lens: LensConfig, r: np.ndarray) -> np.ndarray:
    c = lens.x_front + lens.radius_front
    return c - np.sqrt(np.maximum(0.0, lens.radius_front * lens.radius_front - r * r))


def _lens_back_x(lens: LensConfig, r: np.ndarray) -> np.ndarray:
    c = lens.x_back - lens.radius_back
    return c + np.sqrt(np.maximum(0.0, lens.radius_back * lens.radius_back - r * r))


def _build_lens_mesh(
    lens: LensConfig,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_idx: int,
    n_theta: int = 56,
    n_radial: int = 24,
    front_tri_ids: List[int] | None = None,
    back_tri_ids: List[int] | None = None,
) -> None:
    r_vals = np.linspace(0.0, lens.aperture_radius, n_radial + 1, dtype=np.float64)
    front_x = _lens_front_x(lens, r_vals)
    back_x = _lens_back_x(lens, r_vals)

    front = np.zeros((n_radial + 1, n_theta, 3), dtype=np.float64)
    back = np.zeros((n_radial + 1, n_theta, 3), dtype=np.float64)

    for j in range(n_theta):
        t = (2.0 * math.pi * j) / float(n_theta)
        ct = math.cos(t)
        st = math.sin(t)
        for i, r in enumerate(r_vals):
            y = r * ct
            z = r * st
            front[i, j] = np.array([front_x[i], y, z], dtype=np.float64)
            back[i, j] = np.array([back_x[i], y, z], dtype=np.float64)

    for i in range(n_radial):
        for j in range(n_theta):
            jp = (j + 1) % n_theta

            f00 = front[i, j]
            f01 = front[i, jp]
            f10 = front[i + 1, j]
            f11 = front[i + 1, jp]

            b00 = back[i, j]
            b01 = back[i, jp]
            b10 = back[i + 1, j]
            b11 = back[i + 1, jp]

            if front_tri_ids is not None:
                front_tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, f00, f11, f10, mat_idx)
            if front_tri_ids is not None:
                front_tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, f00, f01, f11, mat_idx)

            if back_tri_ids is not None:
                back_tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, b00, b10, b11, mat_idx)
            if back_tri_ids is not None:
                back_tri_ids.append(len(tri_list))
            _append_tri(tri_list, mat_ids, b00, b11, b01, mat_idx)

    i = n_radial
    for j in range(n_theta):
        jp = (j + 1) % n_theta
        f0 = front[i, j]
        f1 = front[i, jp]
        b0 = back[i, j]
        b1 = back[i, jp]
        _append_tri(tri_list, mat_ids, f0, b0, b1, mat_idx)
        _append_tri(tri_list, mat_ids, f0, b1, f1, mat_idx)


def _build_scene_mesh(
    scene: SceneConfig,
    sidecar: FreeFrequencySidecar,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MaterialDatabase, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, float, float]], np.ndarray, np.ndarray, np.ndarray]:
    db = MaterialDatabase()

    idx_black = db.register(
        "calib_black",
        Material(
            name="calib_black",
            domain="em_optical",
            albedo=[0.0, 0.0, 0.0],
            roughness=0.45,
            metallic=0.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.0,
            transmission=0.0,
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.05,
                transmittance=0.0,
                diffuse_frac=1.0,
                emission_scale=0.0,
                ior_real=1.0,
                ior_imag=0.0,
            ),
        ),
    )
    idx_aperture_black = db.register(
        "calib_aperture_black",
        Material(
            name="calib_aperture_black",
            domain="em_optical",
            albedo=[0.0, 0.0, 0.0],
            roughness=1.0,
            metallic=0.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.0,
            transmission=0.0,
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.0,
                transmittance=0.0,
                diffuse_frac=1.0,
                emission_scale=0.0,
                ior_real=1.0,
                ior_imag=0.0,
            ),
        ),
    )
    idx_lens = db.register(
        "calib_lens_glass",
        Material(
            name="calib_lens_glass",
            domain="em_optical",
            albedo=[1.0, 1.0, 1.0],
            roughness=0.0,
            metallic=0.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=float(scene.lens.ior),
            transmission=0.98,
            spectral_bands=_make_dispersive_lens_bands(
                sidecar,
                base_ior=float(scene.lens.ior),
                reflectance=0.02,
                transmittance=0.98,
            ),
        ),
    )
    idx_source = db.register(
        "calib_source",
        Material(
            name="calib_source",
            domain="em_optical",
            albedo=[1.0, 1.0, 1.0],
            roughness=0.0,
            metallic=0.0,
            emission_rgb=[1.0, 1.0, 1.0],
            ior=1.0,
            transmission=0.0,
            radiance=RadianceProfile(
                luminance=5000.0,
                cct_k=5500.0,
                cri=95.0,
                solid_angle_sr=math.pi * 2.0,
                distribution="lambertian",
            ),
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.0,
                transmittance=0.0,
                diffuse_frac=0.0,
                emission_scale=float(scene.side_room_source_emission),
                ior_real=1.0,
                ior_imag=0.0,
            ),
        ),
    )
    idx_art = db.register(
        "calib_art_mesh",
        Material(
            name="calib_art_mesh",
            domain="em_optical",
            albedo=[0.88, 0.63, 0.34],
            roughness=0.18,
            metallic=0.15,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.46,
            transmission=0.0,
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.55,
                transmittance=0.0,
                diffuse_frac=0.32,
                emission_scale=0.0,
                ior_real=1.46,
                ior_imag=0.0,
            ),
        ),
    )
    # Silver mirror: high specular reflectance, nearly zero diffuse, no transmission.
    # ior_real=1.0 keeps mat_transmissive=False; ior_imag=3.0 gives ~120° Fresnel phase
    # matching silver at visible wavelengths.
    idx_silver = db.register(
        "silver_mirror",
        Material(
            name="silver_mirror",
            domain="em_optical",
            albedo=[0.97, 0.97, 0.97],
            roughness=0.01,
            metallic=1.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.0,
            transmission=0.0,
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.97,
                transmittance=0.0,
                diffuse_frac=0.01,
                emission_scale=0.0,
                ior_real=1.0,
                ior_imag=3.0,
            ),
        ),
    )
    idx_diffuser = db.register(
        "light_room_diffuser",
        Material(
            name="light_room_diffuser",
            domain="em_optical",
            albedo=[0.98, 0.98, 0.98],
            roughness=0.55,
            metallic=0.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.25,
            transmission=float(np.clip(scene.side_room_diffuser_transmittance, 0.0, 1.0)),
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=float(np.clip(1.0 - scene.side_room_diffuser_transmittance, 0.0, 1.0)),
                transmittance=float(np.clip(scene.side_room_diffuser_transmittance, 0.0, 1.0)),
                diffuse_frac=float(np.clip(scene.side_room_diffuser_diffuse_frac, 0.0, 1.0)),
                emission_scale=0.0,
                ior_real=1.25,
                ior_imag=0.0,
            ),
        ),
    )
    idx_stage_grey = db.register(
        "stage_calibration_grey",
        Material(
            name="stage_calibration_grey",
            domain="em_optical",
            albedo=[float(scene.stage_grey_reflectance)] * 3,
            roughness=0.65,
            metallic=0.0,
            emission_rgb=[0.0, 0.0, 0.0],
            ior=1.0,
            transmission=0.0,
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=float(np.clip(scene.stage_grey_reflectance, 0.0, 1.0)),
                transmittance=0.0,
                diffuse_frac=1.0,
                emission_scale=0.0,
                ior_real=1.0,
                ior_imag=0.0,
            ),
        ),
    )

    tris: List[np.ndarray] = []
    mats: List[int] = []
    lens_front_tri_ids: List[int] = []
    lens_back_tri_ids: List[int] = []
    # Per-lens (front_ids, back_ids, radius_front, radius_back) for parametric surface registration.
    lens_surface_groups: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
    silver_wall_tri_ids: List[int] = []
    black_wall_tri_ids: List[int] = []
    camera_body_tri_ids: List[int] = []
    aperture_stop_tri_ids: List[int] = []
    image_plate_tri_ids: List[int] = []
    object_tri_ids: List[int] = []
    lenses: List[LensConfig] = []
    if not bool(getattr(scene, "disable_optics", False)):
        lenses = [l for l in _scene_lenses(scene) if _lens_is_valid(l)]
        if not lenses:
            lenses = [scene.lens]
        
        # Optional macro supplementary lens: inserted at front to extend working distance
        # for extreme close focus.  This enables macro imaging without changing the
        # main optical train; it acts as a magnifying attachment at the object side.
        if getattr(scene, "macro_lens", None) is not None:
            macro = scene.macro_lens
            if _lens_is_valid(macro):
                lenses.insert(0, macro)
                print(
                    "[macro-lens]",
                    f"enabled=1",
                    f"center_x={macro.center_x:.4f}",
                    f"aperture={macro.aperture_radius*1e3:.2f}mm",
                    f"ior={macro.ior:.3f}",
                    flush=True,
                )
            else:
                print("[macro-lens] config invalid; skipping", flush=True)

    # Track source triangles so they can be suppressed from the surface tone map.
    source_tri_ids: List[int] = []
    stage_light_cutters: List[PipeCSGSpec] = []

    # Default legacy side-room tube as first macro light tube.
    default_tube = StageLightTubeConfig(
        opening_x=float(scene.side_room_source_x),
        opening_y=-float(scene.tube_radius),
        opening_z=float(scene.side_room_source_z),
        axis_x=0.0,
        axis_y=-1.0,
        axis_z=0.0,
        depth=float(max(0.02, scene.side_room_depth)),
        profile="square",
        bore_radius=float(scene.side_room_bore_radius),
        n_sides=4,
        diffuser_enabled=bool(scene.side_room_diffuser_enabled),
        diffuser_radius=float(scene.side_room_diffuser_radius),
        diffuser_thickness=float(scene.side_room_diffuser_thickness),
        diffuser_transmittance=float(scene.side_room_diffuser_transmittance),
        diffuser_diffuse_frac=float(scene.side_room_diffuser_diffuse_frac),
        emitter_radius=float(scene.side_room_source_radius),
        emitter_depth_frac=0.90,
    )
    all_tubes = [default_tube] + list(scene.stage_light_tubes)
    for tube_cfg in all_tubes:
        cut_spec = _build_stage_light_tube(
            tube_cfg,
            idx_wall=idx_silver,
            idx_diffuser=idx_diffuser,
            idx_source=idx_source,
            tri_list=tris,
            mat_ids=mats,
            source_tri_ids=source_tri_ids,
        )
        stage_light_cutters.append(cut_spec)

    if bool(scene.stage_probe_emitter_enabled):
        _build_emissive_sphere(
            center=np.array([
                float(scene.stage_probe_x),
                float(scene.stage_probe_y),
                float(scene.stage_probe_z),
            ], dtype=np.float64),
            radius=float(max(1.0e-4, scene.stage_probe_radius)),
            tri_list=tris,
            mat_ids=mats,
            mat_idx=idx_source,
            n_theta=20,
            n_phi=12,
            tri_ids=source_tri_ids,
        )
        print(
            "[stage-probe]",
            "enabled=1",
            f"center=({scene.stage_probe_x:.3f},{scene.stage_probe_y:.3f},{scene.stage_probe_z:.3f})",
            f"radius={scene.stage_probe_radius:.3f}",
            flush=True,
        )
    
    # ─── Diagnostic: Diffuser Material Verification ─────────────────────────────
    diffuser_mat = db._materials.get("light_room_diffuser")
    if diffuser_mat:
        print(
            "[diffuser-material]",
            f"transmission={diffuser_mat.transmission}",
            f"transmittance_config={scene.side_room_diffuser_transmittance}",
            f"albedo={diffuser_mat.albedo}",
            f"roughness={diffuser_mat.roughness}",
            flush=True,
        )
    else:
        print("[diffuser-material] NOT FOUND in database", flush=True)
    
    print(
        "[macro-stage-light]",
        f"tubes={len(all_tubes)}",
        f"open_mouth={1 if scene.side_room_diffuser_enabled else 0}",
        f"diffuser_mouth={1 if scene.side_room_diffuser_enabled else 0}",
        flush=True,
    )
    
    _build_decorative_mesh(
        tris,
        mats,
        idx_art,
        center=np.array([scene.object_plane.x + 0.24, 0.0, 0.0], dtype=np.float64),
        radius=0.045,
        n_u=40,
        n_v=28,
        tri_ids=object_tri_ids,
    )

    # Mirror end-cap disc at the back of the light chamber (at source_x, faces +x inward).
    _build_disc_cap(scene.source_x, scene.tube_radius, tris, mats, idx_silver, n_theta=48)

    # Light chamber entrance: silver mirror cylinder from source to the
    # collimating baffle.
    _build_cylinder_walls(
        scene.source_x,
        scene.baffle0_x,
        scene.tube_radius,
        64,
        tris,
        mats,
        idx_silver,
        tri_ids=silver_wall_tri_ids,
    )
    _iris = getattr(scene, "iris_aperture", None)
    _iris_active = _iris is not None and getattr(_iris, "enabled", False)
    if _iris_active:
        _build_iris_baffle(_iris, scene.tube_radius, tris, mats, idx_aperture_black, aperture_stop_tri_ids)
    if lenses:
        first_lens = lenses[0]
        last_lens = lenses[-1]
        # Place the exit pupil / aperture stop just behind the last lens so
        # sensor backcast rays don't have to pass through a long barrel before
        # reaching it.  Only override when the scene still holds the dataclass
        # default (1.85); explicit non-default values are respected as-is.
        if abs(scene.exit_pupil_x - 1.85) < 1.0e-4:
            scene.exit_pupil_x = round(float(last_lens.x_back) + 0.015, 6)
        # Pre-lens bore: narrow silver sleeve from baffle0 to lens front.
        # This keeps rays confined and prevents pre-lens spray.
        # Has a hole-saw opening where the side-room light chamber penetrates.
        # The stage chamber stays at the configured chamber radius. Narrowing
        # this segment creates an artificial annular discontinuity at baffle0_x.
        bore_r = float(scene.tube_radius)
        # General CSG opening: subtract all stage-light tube interiors from the
        # main pre-lens bore wall so each tube forms an interior union channel.
        host_pipe = PipeCSGSpec(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            t0=float(scene.baffle0_x),
            t1=float(first_lens.x_front - 0.002),
            kind="circle",
            radius=float(bore_r),
        )
        _build_pipe_walls_csg(
            host_pipe,
            tris,
            mats,
            idx_silver,
            n_axial=64,
            n_profile=64,
            cutters=stage_light_cutters,
            csg_mode="subtract",
            tri_ids=silver_wall_tri_ids,
        )
        # Lens housing sleeve: keeps the lens mechanically inset in a bore rather
        # than visually floating in open space.
        # Built with CSG so stage light tube bore holes are subtracted, allowing
        # diffuse light to escape from the diffuser into the interior.
        # Tighten the housing bore to last lens aperture to prevent rear light escape.
        lens_housing_r = min(scene.tube_radius, last_lens.aperture_radius + 0.008)
        lens_housing_pipe = PipeCSGSpec(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            t0=float(first_lens.x_front - 0.002),
            t1=float(last_lens.x_back + 0.002),
            kind="circle",
            radius=float(lens_housing_r),
        )
        _build_pipe_walls_csg(
            lens_housing_pipe,
            tris,
            mats,
            idx_stage_grey,
            n_axial=64,
            n_profile=64,
            cutters=stage_light_cutters,
            csg_mode="subtract",
            tri_ids=black_wall_tri_ids,
        )
        # Lens-edge sealing lip: slightly overlaps the lens edge to prevent a
        # grazing light skirt from bypassing around the lens perimeter.
        # Tighter inner edge seal to prevent light leaks.
        lens_lip_inner_r = max(0.0, min(scene.tube_radius - 1.0e-4, first_lens.aperture_radius - 0.001))
        # Lens seal lip (front): hard-stop perimeter bypass around lens edge.
        _build_baffle_annulus(first_lens.x_front - 0.002, lens_lip_inner_r, scene.tube_radius, 72, tris, mats, idx_black, tri_ids=None)
        for lens in lenses:
            # Tighter clearances on per-lens mechanical stage to seal light escape paths.
            stage_clear_r = min(scene.tube_radius - 1.0e-4, lens.aperture_radius + 0.004)
            stage_lip_r = max(0.0, min(scene.tube_radius - 1.0e-4, lens.aperture_radius - 0.0008))
            # Per-lens mechanical stage: seat + front/back sealing flanges.
            _build_baffle_annulus(lens.x_front - 0.010, stage_clear_r, scene.tube_radius, 72, tris, mats, idx_black, tri_ids=None)
            _build_baffle_annulus(lens.x_front - 0.002, stage_lip_r, scene.tube_radius, 72, tris, mats, idx_black, tri_ids=None)
            _build_baffle_annulus(lens.x_back + 0.002, stage_lip_r, scene.tube_radius, 72, tris, mats, idx_black, tri_ids=None)
            _build_baffle_annulus(lens.x_back + 0.010, stage_clear_r, scene.tube_radius, 72, tris, mats, idx_black, tri_ids=None)
            _f_start = len(lens_front_tri_ids)
            _b_start = len(lens_back_tri_ids)
            _build_lens_mesh(
                lens,
                tris,
                mats,
                idx_lens,
                n_theta=64,
                n_radial=24,
                front_tri_ids=lens_front_tri_ids,
                back_tri_ids=lens_back_tri_ids,
            )
            lens_surface_groups.append((
                np.ascontiguousarray(lens_front_tri_ids[_f_start:], dtype=np.int32),
                np.ascontiguousarray(lens_back_tri_ids[_b_start:], dtype=np.int32),
                float(lens.radius_front),
                float(lens.radius_back),
            ))

        # Post-lens bellows frustum: flares from the last-lens aperture radius
        # out to the full sensor plate radius.  No cylindrical tube — every pixel
        # on the sensor has line-of-sight back through the lens with no wall strike.
        _pl_x0 = float(last_lens.x_back + 0.002)
        _pl_x1 = float(scene.image_plate.x - 0.002)
        _pl_r0 = float(last_lens.aperture_radius)
        _pl_r1 = float(scene.image_plate.radius)
        _build_sensor_aperture_frustum(
            _pl_x0, _pl_r0, _pl_x1, _pl_r1,
            96, tris, mats, idx_aperture_black, tri_ids=black_wall_tri_ids,
        )
    else:
        # No-lens path: interior from baffle0 to image plate, subtracted by light tubes.
        baffle_to_image_pipe = PipeCSGSpec(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            axis_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            t0=float(scene.baffle0_x),
            t1=float(scene.image_plate.x - 0.002),
            kind="circle",
            radius=float(scene.tube_radius),
        )
        _build_pipe_walls_csg(
            baffle_to_image_pipe,
            tris,
            mats,
            idx_stage_grey,
            n_axial=64,
            n_profile=64,
            cutters=stage_light_cutters,
            csg_mode="subtract",
            tri_ids=black_wall_tri_ids,
        )
    
    # Exit pupil / field stop aperture: limits light transmission post-optics.
    _build_baffle_annulus(
        float(scene.exit_pupil_x),
        float(scene.exit_pupil_radius),
        scene.tube_radius,
        72,
        tris,
        mats,
        idx_aperture_black,
        tri_ids=None,
        thickness=float(scene.exit_pupil_thickness),
    )
    
    _build_screen(scene, tris, mats, idx_black, tri_ids=image_plate_tri_ids)

    # Sensor chamber enclosure around the full circular sensor plate.  The
    # outer wall tapers from the post-lens tube to the sensor disc instead of
    # carrying the sensor radius all the way back to the exit-pupil plane.
    # Without this enclosure every backward ray from an outer pixel launches
    # into open air and traverses the entire scene unchecked.
    #
    # Three pieces:
    #   1. Outer tapered hull — lateral wall, exit_pupil_x → image_plate.x
    #   2. Rear annular cap   — seals face at exit_pupil_x (tube_r → rear hull)
    #   3. Front annular cap  — seals disc-vs-hull gap at image_plate.x
    _x_ap      = float(scene.exit_pupil_x)
    _x_sensor  = float(scene.image_plate.x)
    _tube_r    = float(scene.tube_radius)
    _sensor_r  = float(scene.image_plate.radius)
    _hull_margin = 0.010
    _hull_r_ap     = _tube_r + _hull_margin
    _hull_r_sensor = _sensor_r + _hull_margin

    # 1. Outer tapered hull — tracked separately so it can have its own UV page.
    camera_barrel_tri_ids: list = []
    _build_sensor_aperture_frustum(
        x_aperture=_x_ap,
        r_aperture=_hull_r_ap,
        x_sensor=_x_sensor,
        r_sensor=_hull_r_sensor,
        n_theta=96,
        tri_list=tris,
        mat_ids=mats,
        mat_idx=idx_aperture_black,
        tri_ids=camera_barrel_tri_ids,
    )
    camera_body_tri_ids.extend(camera_barrel_tri_ids)
    # 2. Rear cap: ring from tube wall to tapered hull (closes the back)
    camera_rear_cap_tri_ids: list = []
    _build_baffle_annulus(
        _x_ap, _tube_r, _hull_r_ap, 96,
        tris, mats, idx_aperture_black,
        tri_ids=camera_rear_cap_tri_ids,
        thickness=0.0,
    )
    camera_body_tri_ids.extend(camera_rear_cap_tri_ids)
    # 3. Front cap: ring from sensor circle to tapered hull
    camera_front_cap_tri_ids: list = []
    _build_baffle_annulus(
        _x_sensor, _sensor_r, _hull_r_sensor, 96,
        tris, mats, idx_aperture_black,
        tri_ids=camera_front_cap_tri_ids,
        thickness=0.0,
    )
    camera_body_tri_ids.extend(camera_front_cap_tri_ids)

    # Sensor-to-aperture frustum: inner cone wall connecting exit-pupil
    # aperture hole to the sensor diagonal so all corner pixels are bounded.
    camera_frustum_tri_ids: list = []
    _build_sensor_aperture_frustum(
        x_aperture=_x_ap,
        r_aperture=float(scene.exit_pupil_radius),
        x_sensor=_x_sensor,
        r_sensor=_sensor_r,
        n_theta=96,
        tri_list=tris,
        mat_ids=mats,
        mat_idx=idx_aperture_black,
        tri_ids=camera_frustum_tri_ids,
    )
    camera_body_tri_ids.extend(camera_frustum_tri_ids)
    print(
        "[sensor-enclosure]",
        f"x_ap={_x_ap:.4f}",
        f"x_sensor={_x_sensor:.4f}",
        f"tube_r={_tube_r*1e3:.1f}mm",
        f"sensor_r={_sensor_r*1e3:.1f}mm",
        f"hull_r=({_hull_r_ap*1e3:.1f},{_hull_r_sensor*1e3:.1f})mm",
        flush=True,
    )

    if scene.enable_debug_plate:
        debug_x = float(scene.debug_plate_x)
        debug_r = float(scene.debug_plate_radius)
        p00 = np.array([debug_x, -debug_r, -debug_r], dtype=np.float64)
        p01 = np.array([debug_x, -debug_r, +debug_r], dtype=np.float64)
        p10 = np.array([debug_x, +debug_r, -debug_r], dtype=np.float64)
        p11 = np.array([debug_x, +debug_r, +debug_r], dtype=np.float64)

        _append_tri(tris, mats, p00, p10, p11, idx_black)
        _append_tri(tris, mats, p00, p11, p01, idx_black)

    # ── Red leak-probe sphere: outside the camera barrel, red-only emitter ──
    # Placed just past the camera's rear edge (x > image_plate_x) and outside
    # the barrel radius, so it only illuminates the sensor if there is a gap
    # in the sealed enclosure.  Any red energy on the sensor = confirmed leak.
    idx_red_probe = db.register(
        "red_leak_probe",
        Material(
            name="red_leak_probe",
            domain="em_optical",
            albedo=[1.0, 0.0, 0.0],
            roughness=0.0,
            metallic=0.0,
            emission_rgb=[1.0, 0.0, 0.0],
            ior=1.0,
            transmission=0.0,
            radiance=RadianceProfile(
                luminance=3000.0,
                cct_k=1800.0,
                cri=20.0,
                solid_angle_sr=math.pi * 2.0,
                distribution="lambertian",
            ),
            spectral_bands=_make_red_only_spectral_bands(
                sidecar,
                emission_scale=float(scene.side_room_source_emission),
            ),
        ),
    )
    _probe_r  = 0.018   # 18 mm radius
    _probe_x  = (float(scene.object_plane.x) + float(scene.screen_x)) * 0.5  # center of display region
    _probe_y  = float(scene.view_radius) * 0.90   # near top, inside the [0,1]³ normalized volume
    _probe_z  = float(scene.view_radius) * 0.90   # near camera side (+Z), inside the volume
    red_probe_tri_ids: list = []
    _build_emissive_sphere(
        center=np.array([_probe_x, _probe_y, _probe_z], dtype=np.float64),
        radius=_probe_r,
        tri_list=tris,
        mat_ids=mats,
        mat_idx=idx_red_probe,
        n_theta=24,
        n_phi=12,
        tri_ids=red_probe_tri_ids,
    )
    print(
        f"[red-probe] x={_probe_x:.4f} y={_probe_y:.4f} z={_probe_z:.4f} r={_probe_r*1e3:.1f}mm"
        f" tris={len(red_probe_tri_ids)} mat_idx={idx_red_probe}",
        flush=True,
    )

    tri_arr = np.ascontiguousarray(np.asarray(tris, dtype=np.float64))
    _orient_surface_patch_outward(tri_arr, lens_front_tri_ids, expected_x_sign=-1.0)
    _orient_surface_patch_outward(tri_arr, lens_back_tri_ids, expected_x_sign=1.0)
    verts = tri_arr.reshape(tri_arr.shape[0], 9)
    normals = np.stack([_normal(t[0], t[1], t[2]) for t in tri_arr], axis=0).astype(np.float64)
    mat_idx = np.ascontiguousarray(np.asarray(mats, dtype=np.int32))

    # Surface overlay suppression: triangles listed here are zeroed out of
    # tri_flux before tone-mapping so they don't swamp the display dynamic
    # range.  Add any always-emissive or structural surface that should be
    # FIELD_EXEMPT (excluded from the field-integration overlay).
    suppress_ids = np.ascontiguousarray(
        np.concatenate([
            np.asarray(source_tri_ids,    dtype=np.int32),  # FIELD_EXEMPT: primary emitter
            np.asarray(red_probe_tri_ids, dtype=np.int32),  # FIELD_EXEMPT: debug probe
            np.asarray(silver_wall_tri_ids, dtype=np.int32),
            np.asarray(black_wall_tri_ids,  dtype=np.int32),
        ]),
        dtype=np.int32,
    )
    tube_baffle_ids = np.ascontiguousarray(
        np.concatenate([
            np.asarray(silver_wall_tri_ids, dtype=np.int32),
            np.asarray(black_wall_tri_ids, dtype=np.int32),
            np.asarray(camera_body_tri_ids, dtype=np.int32),
        ]),
        dtype=np.int32,
    )
    return (
        verts,
        normals,
        mat_idx,
        tri_arr,
        db,
        np.ascontiguousarray(np.asarray(source_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(lens_front_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(lens_back_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(image_plate_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(aperture_stop_tri_ids, dtype=np.int32)),
        suppress_ids,
        lens_surface_groups,
        np.ascontiguousarray(np.asarray(object_tri_ids, dtype=np.int32)),
        tube_baffle_ids,
        np.ascontiguousarray(np.asarray(camera_barrel_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(camera_rear_cap_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(camera_front_cap_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(camera_frustum_tri_ids, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(red_probe_tri_ids, dtype=np.int32)),
    )


@dataclass
class UvSurfaceGroup:
    name: str
    kind: str
    tri_ids: np.ndarray
    uv_coords: np.ndarray
    res: int = UV_PAGE_RES_DEFAULT
    layer: int = -1
    group_id: int = -1
    hot: bool = True
    warm_channels: Optional[np.ndarray] = None
    warm_page: Optional[np.ndarray] = None
    last_summary: Dict[str, object] = field(default_factory=dict)


class UvPageBank:
    """Physical-surface UV page registry and warm-page cache."""

    def __init__(self, n_bands: int, res: int = UV_PAGE_RES_DEFAULT,
                 hot_limit: int = UV_HOT_GROUP_LIMIT_DEFAULT) -> None:
        self.n_bands = int(n_bands)
        self.res = int(res)
        self.hot_limit = int(hot_limit)
        self.groups: List[UvSurfaceGroup] = []
        self._by_gid: Dict[int, UvSurfaceGroup] = {}

    @property
    def n_channels(self) -> int:
        return UV_HDR_CHANNELS + 5 * self.n_bands

    def add(self, name: str, kind: str, tri_ids: np.ndarray, uv_coords: np.ndarray) -> None:
        ids = np.ascontiguousarray(np.asarray(tri_ids, dtype=np.int32).reshape(-1), dtype=np.int32)
        if ids.size <= 0:
            return
        uv = np.ascontiguousarray(np.asarray(uv_coords, dtype=np.float32).reshape(ids.size, 3, 2), dtype=np.float32)
        layer = len(self.groups)
        self.groups.append(UvSurfaceGroup(
            name=str(name),
            kind=str(kind),
            tri_ids=ids,
            uv_coords=uv,
            res=self.res,
            layer=layer,
            hot=(layer < self.hot_limit),
        ))

    def register_all(self, tracer, sample_area: int) -> None:
        self._by_gid.clear()
        for g in self.groups:
            g.group_id = -1
            if g.warm_page is None:
                g.warm_page = np.zeros((g.res, g.res, 4), dtype=np.float32)
            if not g.hot:
                continue
            gid = int(tracer.register_tri_group(
                0,
                int(sample_area),
                np.ascontiguousarray(g.tri_ids, dtype=np.int32),
                uv_image={
                    "res": int(g.res),
                    "uv_coords": np.ascontiguousarray(g.uv_coords, dtype=np.float32),
                },
            ))
            g.group_id = gid
            self._by_gid[gid] = g
            if g.warm_channels is not None:
                tracer.set_group_uv_image(
                    gid,
                    np.ascontiguousarray(g.warm_channels, dtype=np.float32),
                )

    def update_from_tracer(self, tracer, freq_hz: np.ndarray, mode: str = "combined") -> np.ndarray:
        layers = max(1, len(self.groups))
        tex = np.zeros((layers, self.res, self.res, 4), dtype=np.float32)
        if not self.groups:
            return tex
        for g in self.groups:
            if g.warm_page is not None:
                tex[g.layer] = np.asarray(g.warm_page, dtype=np.float32)
        wl_nm = (C_LIGHT / np.maximum(np.asarray(freq_hz, dtype=np.float64)[:self.n_bands], EPS)) * 1.0e9
        rgb_w = _wavelength_to_rgb_weights(wl_nm).astype(np.float32)
        for g in self.groups:
            if g.group_id < 0:
                continue
            try:
                d = tracer.get_group_uv_image(int(g.group_id))
                summary = tracer.get_group_uv_summary(int(g.group_id))
            except Exception as exc:
                if not g.last_summary.get("warned"):
                    print(f"[uv-bank] readback unavailable group={g.name} gid={g.group_id}: {exc}", flush=True)
                    g.last_summary["warned"] = True
                continue
            channels = np.asarray(d["channels"], dtype=np.float32)
            nb = int(d.get("n_bands", self.n_bands))
            g.warm_channels = np.ascontiguousarray(channels, dtype=np.float32)
            if channels.shape[0] >= UV_HDR_CHANNELS + 5 * nb:
                fwd = channels[UV_HDR_CHANNELS + 3 * nb:UV_HDR_CHANNELS + 4 * nb]
                sen = channels[UV_HDR_CHANNELS + 4 * nb:UV_HDR_CHANNELS + 5 * nb]
                if mode == "forward":
                    band_mags = fwd
                elif mode == "sensor":
                    band_mags = sen
                elif mode == "difference":
                    band_mags = np.maximum(fwd - sen, 0.0)
                else:
                    band_mags = fwd + sen
            else:
                band_mags = channels[UV_HDR_CHANNELS:UV_HDR_CHANNELS + nb]
            rgb = np.einsum("byx,bc->yxc", band_mags[:self.n_bands], rgb_w[:band_mags.shape[0]], optimize=True)
            alpha = np.maximum.reduce(rgb, axis=2) if rgb.size else np.zeros((self.res, self.res), dtype=np.float32)
            tex[g.layer, :, :, :3] = np.maximum(rgb, 0.0)
            tex[g.layer, :, :, 3] = np.maximum(alpha, 0.0)
            g.warm_page = tex[g.layer].copy()
            g.last_summary = dict(summary)
        return tex

    def metadata(self) -> List[Dict[str, object]]:
        return [
            {
                "name": g.name,
                "kind": g.kind,
                "group_id": int(g.group_id),
                "layer": int(g.layer),
                "res": int(g.res),
                "hot": bool(g.hot),
                "tri_count": int(g.tri_ids.size),
                **g.last_summary,
            }
            for g in self.groups
        ]

    def refresh_summaries(self, tracer) -> bool:
        any_nonzero = False
        for g in self.groups:
            if g.group_id < 0:
                continue
            try:
                s = tracer.get_group_uv_summary(int(g.group_id))
            except Exception:
                continue
            g.last_summary = dict(s)
            nz = int(g.last_summary.get("nonzero_texels", 0) or 0)
            fwd = float(g.last_summary.get("total_forward", 0.0) or 0.0)
            sen = float(g.last_summary.get("total_sensor", 0.0) or 0.0)
            any_nonzero = any_nonzero or (nz > 0 or fwd > 0.0 or sen > 0.0)
        return any_nonzero

    def any_hot_data(self) -> bool:
        for g in self.groups:
            if not g.hot:
                continue
            s = g.last_summary
            if int(s.get("nonzero_texels", 0) or 0) > 0:
                return True
            if float(s.get("total_forward", 0.0) or 0.0) > 0.0:
                return True
            if float(s.get("total_sensor", 0.0) or 0.0) > 0.0:
                return True
        return False

    def memory_report(self) -> Dict[str, float]:
        group_count = len(self.groups)
        hot_count = sum(1 for g in self.groups if g.hot)
        accum_bytes = self.n_channels * self.res * self.res * 4
        warm_bytes = group_count * self.res * self.res * 4 * 4
        return {
            "groups": float(group_count),
            "hot_groups": float(hot_count),
            "channels": float(self.n_channels),
            "gpu_hot_mb": float(hot_count * accum_bytes) / (1024.0 * 1024.0),
            "planned_warm_analytical_mb": float(group_count * accum_bytes) / (1024.0 * 1024.0),
            "warm_rgba32f_mb": float(warm_bytes) / (1024.0 * 1024.0),
        }


@dataclass
class ForwardCppLensBench:
    scene: SceneConfig
    freq_hz: np.ndarray
    view_h: int
    view_w: int
    sidecar: FreeFrequencySidecar | None = None
    field_capture: bool = True

    def __post_init__(self) -> None:
        self._trace_lock = threading.Lock()
        self.bdpt_consecutive_no_survivor_frames = 0
        self.bdpt_debug_print_counter = 0
        self._bdpt_sensor_cfg: Tuple[int, int] | None = None
        self._bdpt_source_gid = -1
        self._bdpt_red_probe_gid = -1
        self.bdpt_last_sensor_gid = -1
        self.uv_page_bank: Optional[UvPageBank] = None
        self.bdpt_last_launched_rays = 0
        self.bdpt_last_records = 0
        self.bdpt_last_volume_records = 0
        self.bdpt_last_volume_power = 0.0
        self.bdpt_cumulative_volume_records = 0
        self.bdpt_cumulative_volume_power = 0.0
        self.bdpt_last_telemetry: Dict[str, int] = {}
        self.bdpt_last_survivor_records = 0
        self.bdpt_last_sensor_photons = 0.0
        self.bdpt_last_sensor_power = 0.0
        self.bdpt_last_n_px = int(max(4, int(self.scene.image_plate.pixels)))
        self.bdpt_last_aperture_samples = 1
        self._surface_top_ema: Optional[np.ndarray] = None
        self._surface_side_ema: Optional[np.ndarray] = None
        self._field_top_ema: Optional[np.ndarray] = None
        self._field_side_ema: Optional[np.ndarray] = None
        self._plate_rgb_accum: Optional[np.ndarray] = None
        self._plate_sensor_photons_accum: Optional[np.ndarray] = None
        self._plate_sensor_electrons_accum: Optional[np.ndarray] = None
        self._last_bdpt_records: Optional[np.ndarray] = None
        # GPU/CPU compute mode: 'gpu', 'cpu', or 'mixed'.
        # Controls use_gpu_compute and gpu_all_stages in submit_rays.
        self.compute_mode: str = "gpu"
        # Persistent-pipeline drain loop
        self._drain_thread: Optional[threading.Thread] = None
        self._drain_stop  = threading.Event()
        self._segs_lock   = threading.Lock()
        # Bounded ring buffer of raw hit positions for the point overlay.
        # Each row: (x, y, z, amplitude, display_class).  Memory is constant:
        # 65536 rows × 5 floats × 4 bytes = 1.25 MB.  No spatial quantization.
        _VIS_CAP          = 2_000_000
        self._VIS_CAP     = _VIS_CAP
        self._vis_buf     = np.zeros((_VIS_CAP, 5), dtype=np.float32)
        self._vis_ptr     = 0     # next write position (ring head)
        self._vis_full    = False  # True once ring has wrapped at least once
        # Tunable amplitude floor: rays (and child spawns) below this threshold
        # are terminated.  Also used for material epsilon-kill pre-flagging.
        self._min_amplitude: float = 1e-5
        # Sensor ray bias/gain: sensor rays are launched with amplitude
        # multiplied by sensor_amp_gain (pre-compensates multi-lens Fresnel
        # dropoff) and survive until amplitude falls below sensor_min_amplitude
        # (0.0 = never kill on amplitude, appropriate for single-photon detectors).
        self.sensor_amp_gain: float = 1.0
        self.sensor_min_amplitude: float = 0.0
        # Emitter ray gain: forward rays are launched with amplitude multiplied
        # by emitter_amp_gain.  Use to pre-compensate scene absorption or boost
        # light level without changing the physical source count.
        self.emitter_amp_gain: float = 1.0
        self._sensor_aim_reported: bool = False
        # Whether the pipeline has been configured with adaptive thresholds yet
        self._pipeline_configured: bool = False
        # Preferred render frame rate (Hz) and blending weight used in drain-loop
        # batch-time targeting.  The drain loop blends between a fixed 20 ms
        # default and a per-frame budget derived from target_fps.
        # fps_alpha = 0.0 → ignore fps entirely; 1.0 → follow fps exactly.
        self.target_fps:  float = 30.0
        self.fps_alpha:   float = 0.35   # moderate pull toward frame-rate awareness
        # Intent-queue shuffle lever: 0.0 = FIFO, 1.0 = fully random window.
        # Diffuses rays across depth and breadth of the BVH traversal tree.
        self.intent_shuffle: float = 0.0
        # Live GPU profiling: snapshots collected during early pipeline work.
        # Once _gpu_calibrated is True, a calibration report has been printed.
        self._gpu_profile_snapshots: list = []
        self._gpu_calibrated: bool = False
        n_req = int(np.asarray(self.freq_hz, dtype=np.float64).size)
        if self.sidecar is None:
            self.sidecar = FreeFrequencySidecar.lazy_prepare(n_req)
        freq_hz_vec = np.ascontiguousarray(np.asarray(self.sidecar.freq_hz, dtype=np.float64).reshape(-1), dtype=np.float64)
        if int(freq_hz_vec.size) != n_req:
            if int(freq_hz_vec.size) > n_req:
                freq_hz_vec = np.ascontiguousarray(freq_hz_vec[:n_req], dtype=np.float64)
            else:
                pad = np.ascontiguousarray(np.asarray(self.freq_hz, dtype=np.float64).reshape(-1), dtype=np.float64)
                if int(pad.size) < n_req:
                    pad = np.pad(pad, (0, n_req - int(pad.size)), mode="edge")
                freq_hz_vec = np.ascontiguousarray(np.concatenate([freq_hz_vec, pad[int(freq_hz_vec.size):n_req]]), dtype=np.float64)
        self.freq_hz = freq_hz_vec

        (
            verts, normals, mat_idx, tri_arr, db,
            source_ids, lens_front_ids, lens_back_ids, image_plate_ids,
            aperture_stop_ids, tube_wall_ids, lens_surface_groups,
            object_ids, tube_baffle_ids, camera_barrel_ids,
            camera_rear_cap_ids, camera_front_cap_ids, camera_frustum_ids,
            red_probe_ids,
        ) = _build_scene_mesh(self.scene, self.sidecar)
        self.lens_surface_groups = lens_surface_groups
        self.tri_vertices = np.ascontiguousarray(tri_arr, dtype=np.float64)
        self.tri_centroids = np.ascontiguousarray(np.mean(tri_arr, axis=1), dtype=np.float64)
        self.n_tris = int(tri_arr.shape[0])
        self.n_bands = int(self.freq_hz.shape[0])
        self.source_tri_ids = np.ascontiguousarray(source_ids, dtype=np.int32)
        self.image_plate_tri_ids = np.ascontiguousarray(image_plate_ids, dtype=np.int32)
        self.aperture_stop_tri_ids = np.ascontiguousarray(aperture_stop_ids, dtype=np.int32)
        self.object_tri_ids = np.ascontiguousarray(object_ids, dtype=np.int32)
        self.tube_baffle_tri_ids = np.ascontiguousarray(tube_baffle_ids, dtype=np.int32)
        self.camera_barrel_tri_ids   = np.ascontiguousarray(camera_barrel_ids,    dtype=np.int32)
        self.camera_rear_cap_tri_ids = np.ascontiguousarray(camera_rear_cap_ids,  dtype=np.int32)
        self.camera_front_cap_tri_ids= np.ascontiguousarray(camera_front_cap_ids, dtype=np.int32)
        self.camera_frustum_tri_ids  = np.ascontiguousarray(camera_frustum_ids,   dtype=np.int32)
        self.red_probe_tri_ids       = np.ascontiguousarray(red_probe_ids,         dtype=np.int32)
        self.emitter_tri_ids = np.ascontiguousarray(
            np.concatenate([self.source_tri_ids, self.red_probe_tri_ids]).astype(np.int32),
            dtype=np.int32,
        )
        # tube_wall_ids = silver + black cylinder walls only (no flat caps or frustum).
        # Used for UV display so the x_cylinder UV mode stays valid (no degenerate flat discs).
        self.tube_wall_tri_ids = np.ascontiguousarray(tube_wall_ids, dtype=np.int32)
        self.material_db  = db
        self.tri_mat_ids  = np.ascontiguousarray(mat_idx, dtype=np.int32)

        # Per-triangle kind for cross-section rendering.
        tri_kind = np.full(self.n_tris, TRI_KIND_DEFAULT, dtype=np.int8)
        tri_kind[np.asarray(lens_front_ids, dtype=np.int32)] = TRI_KIND_LENS
        tri_kind[np.asarray(lens_back_ids,  dtype=np.int32)] = TRI_KIND_LENS
        tri_kind[np.asarray(aperture_stop_ids, dtype=np.int32)] = TRI_KIND_APERTURE
        tri_kind[np.asarray(source_ids,     dtype=np.int32)] = TRI_KIND_EMISSIVE
        tri_kind[np.asarray(image_plate_ids, dtype=np.int32)] = TRI_KIND_SENSOR
        self.tri_kind = tri_kind
        
        # Build material buffer early so we can extract indices for diagnostics
        mat_buf = db.build_mat_buf().astype(np.float32, copy=False)
        mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)

        # ─── Diagnostic: Material Buffer Verification ─────────────────────────
        tensors = db.build_tensors()
        mat_names = list(tensors.get('index', {}).keys())
        self._mat_index = tensors.get('index', {})
        mat_name_by_idx = {int(v): str(k) for k, v in self._mat_index.items()}
        diffuser_idx = self._mat_index.get('light_room_diffuser', -1)
        self._diffuser_mat_idx = diffuser_idx
        stage_grey_idx = self._mat_index.get('stage_calibration_grey', -1)
        self._stage_grey_mat_idx = stage_grey_idx
        silver_idx = self._mat_index.get('silver_mirror', -1)
        aperture_black_idx = self._mat_index.get('calib_aperture_black', -1)
        
        if diffuser_idx >= 0 and diffuser_idx < len(mat_names):
            mat_row = mat_buf[diffuser_idx * MAX_SPECTRAL_BANDS : (diffuser_idx + 1) * MAX_SPECTRAL_BANDS]
            if mat_row.shape[0] > 0:
                # Spectral band layout: [center_hz, bandwidth_hz, reflectance, transmittance, ...]
                trans_band0 = float(mat_row[0, 3]) if mat_row.shape[1] > 3 else 0.0
                refl_band0 = float(mat_row[0, 2]) if mat_row.shape[1] > 2 else 0.0
                print(
                    "[diffuser-buffer]",
                    f"idx={diffuser_idx}",
                    f"transmittance[band0]={trans_band0:.4f}",
                    f"reflectance[band0]={refl_band0:.4f}",
                    f"config_transmittance={self.scene.side_room_diffuser_transmittance:.4f}",
                    flush=True,
                )
        
        if silver_idx >= 0:
            mat_row = mat_buf[silver_idx * MAX_SPECTRAL_BANDS : (silver_idx + 1) * MAX_SPECTRAL_BANDS]
            if mat_row.shape[0] > 0:
                refl_band0 = float(mat_row[0, 2]) if mat_row.shape[1] > 2 else 0.0
                trans_band0 = float(mat_row[0, 3]) if mat_row.shape[1] > 3 else 0.0
                ior_real = float(mat_row[0, 7]) if mat_row.shape[1] > 7 else 1.0
                ior_imag = float(mat_row[0, 8]) if mat_row.shape[1] > 8 else 0.0
                print(
                    "[silver-mirror-buffer]",
                    f"idx={silver_idx}",
                    f"reflectance[band0]={refl_band0:.4f}",
                    f"transmittance[band0]={trans_band0:.4f}",
                    f"ior_real={ior_real:.4f}",
                    f"ior_imag={ior_imag:.4f}",
                    flush=True,
                )
        print(
            "[materials-registered]",
            f"n_total={len(mat_names)}",
            f"names={mat_names[:8]}...",
            f"mat_rows={mat_n_mats}",
            flush=True,
        )
        
        # ─── Diagnostic: Triangle count by material ─────────────────────────────
        tri_count_by_mat = {}
        for m_idx in mat_idx:
            tri_count_by_mat[int(m_idx)] = tri_count_by_mat.get(int(m_idx), 0) + 1
        mat_idx_arr = np.asarray(mat_idx, dtype=np.int32)
        print(
            "[tris-by-material]",
            f"total_tris={len(mat_idx)}",
            f"diffuser_tris={tri_count_by_mat.get(self._diffuser_mat_idx, 0)}",
            f"stage_grey_tris={tri_count_by_mat.get(self._stage_grey_mat_idx, 0)}",
            flush=True,
        )
        plate_mat_ids = np.unique(mat_idx_arr[self.image_plate_tri_ids]).astype(np.int32, copy=False) if self.image_plate_tri_ids.size > 0 else np.zeros((0,), dtype=np.int32)
        plate_mat_names = [mat_name_by_idx.get(int(mid), f"<unknown:{int(mid)}>") for mid in plate_mat_ids]
        print(
            "[sensor-plate]",
            f"plate_tris={int(self.image_plate_tri_ids.size)}",
            f"material_ids={plate_mat_ids.tolist()}",
            f"material_names={plate_mat_names}",
            flush=True,
        )
        camera_body_tris = 0
        if aperture_black_idx >= 0:
            cam_mask = (
                (mat_idx_arr == int(aperture_black_idx))
                & (self.tri_vertices[:, :, 0].max(axis=1) >= float(self.scene.exit_pupil_x) - 1.0e-9)
                & (self.tri_vertices[:, :, 0].min(axis=1) <= float(self.scene.image_plate.x) + 1.0e-9)
            )
            camera_body_tris = int(np.count_nonzero(cam_mask))
        print(
            "[camera-body]",
            f"sim_tris={camera_body_tris}",
            f"x=({self.scene.exit_pupil_x:.4f},{self.scene.image_plate.x:.4f})",
            f"material=calib_aperture_black",
            flush=True,
        )
        print(
            "[aperture-stop]",
            f"stop_tris={int(self.aperture_stop_tri_ids.size)}",
            flush=True,
        )
        # Aperture focal region — centroid + bounding radius of stop triangles.
        # Used by sensor-cast to aim rays toward the aperture instead of uniform hemisphere.
        if int(self.aperture_stop_tri_ids.size) > 0:
            ap_cents = self.tri_centroids[self.aperture_stop_tri_ids]   # (K, 3)
            self.aperture_centroid = ap_cents.mean(axis=0).astype(np.float64)
            # The iris triangles span the full opaque area (r_inner to tube_radius),
            # so max-centroid-distance gives the *opaque* footprint radius (~49 mm),
            # not the clear aperture hole.  Use the clear aperture radius instead so
            # sensor-cast stencils are aimed through the exit pupil, not at the blades.
            _iris_cfg = getattr(self.scene, "iris_aperture", None)
            _ep_r     = float(getattr(self.scene, "exit_pupil_radius", 0.0))
            if _iris_cfg is not None and bool(getattr(_iris_cfg, "enabled", False)):
                _iris_r = float(getattr(_iris_cfg, "r_inner", 0.0))
                self.aperture_radius = (min(_iris_r, _ep_r) if _ep_r > 0.0 else _iris_r)
            else:
                self.aperture_radius = float(
                    np.max(np.linalg.norm(ap_cents - self.aperture_centroid, axis=1))
                ) + 1e-4
            # Normal: aperture faces along the optical (X) axis in this scene layout
            self.aperture_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            # No iris configured — target the exit pupil so sensor-cast rays pass
            # through it by construction (aimed directly at that disk).  Rays aimed
            # at the first lens instead get blocked by the exit-pupil baffle first.
            _ep_x = float(getattr(self.scene, "exit_pupil_x", 0.0))
            _ep_r = float(getattr(self.scene, "exit_pupil_radius", 0.0))
            if _ep_r > 0.0 and _ep_x > 0.0:
                self.aperture_centroid = np.array([_ep_x, 0.0, 0.0], dtype=np.float64)
                self.aperture_radius   = _ep_r
            else:
                self.aperture_centroid = np.zeros(3, dtype=np.float64)
                self.aperture_radius   = 0.0
            self.aperture_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        print(
            "[aperture-focal-region]",
            f"centroid={self.aperture_centroid.tolist()}",
            f"radius={self.aperture_radius:.4f}",
            flush=True,
        )
        
        # Suppress walls, lens faces, and FIELD_EXEMPT sources from the surface
        # overlay so the tone-map white point is set by downstream surfaces.
        self._surface_suppress_tri_ids = np.ascontiguousarray(
            np.concatenate([
                tube_wall_ids,
                lens_front_ids,
                lens_back_ids,
                self.red_probe_tri_ids,  # FIELD_EXEMPT: always-emissive debug probe
            ]),
            dtype=np.int32,
        )
        self.lens_normal_report = {
            "front_mean_x": float(np.mean(normals[lens_front_ids, 0])) if lens_front_ids.size > 0 else 0.0,
            "back_mean_x": float(np.mean(normals[lens_back_ids, 0])) if lens_back_ids.size > 0 else 0.0,
        }

        mat_buf = db.build_mat_buf().astype(np.float32, copy=False)
        mat_n_mats = int(mat_buf.shape[0] // MAX_SPECTRAL_BANDS)

        self.tracer = _sk.RayTracer(
            n_tri=int(verts.shape[0]),
            verts=np.ascontiguousarray(verts, dtype=np.float64),
            normals=np.ascontiguousarray(normals, dtype=np.float64),
            mat_idx=np.ascontiguousarray(mat_idx, dtype=np.int32),
            mat_buf=np.ascontiguousarray(mat_buf, dtype=np.float32),
            mat_n_mats=mat_n_mats,
            freq_hz=np.ascontiguousarray(self.freq_hz, dtype=np.float64),
            speed_m_s=float(C_LIGHT),
            atmo_abs=np.zeros_like(self.freq_hz, dtype=np.float64),
        )
        self._configure_sensor_film_pipeline()

        # Drive forward tracing from all emissive geometry: primary source + red probe.
        self.src_pos = np.ascontiguousarray(self.tri_centroids[self.emitter_tri_ids], dtype=np.float64)
        src_n = int(self.src_pos.shape[0])
        src_normals = np.ascontiguousarray(normals[self.emitter_tri_ids], dtype=np.float64)
        self.src_dir = np.ascontiguousarray(src_normals, dtype=np.float64)
        # Neutral launch profile: no Python-side beaming. Keep transport driven
        # by emissive materials and scene geometry only.
        self.src_directivity = np.ones((src_n,), dtype=np.float64)
        self.tri_flux = np.zeros((self.n_tris, self.n_bands), dtype=np.float32)
        print(
            "[emitter-tris]",
            f"source={self.source_tri_ids.size}",
            f"red_probe={self.red_probe_tri_ids.size}",
            f"total={self.emitter_tri_ids.size}",
            flush=True,
        )
        print(
            "[lighting-mode] emissive-material-only",
            f"emitter_tris={src_n}",
            "python_power_override=OFF",
            "python_beam_bias=OFF",
            flush=True,
        )
        src_center = np.mean(self.src_pos, axis=0) if src_n > 0 else np.zeros((3,), dtype=np.float64)
        radial = self.src_pos - src_center[None, :]
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_unit = radial / np.maximum(radial_norm, 1.0e-12)
        radial_dot = np.sum(radial_unit * self.src_dir, axis=1) if src_n > 0 else np.zeros((0,), dtype=np.float64)
        print(
            "[source-normal-check]",
            f"mean_dot={float(np.mean(radial_dot)) if radial_dot.size else 0.0:+.4f}",
            f"min_dot={float(np.min(radial_dot)) if radial_dot.size else 0.0:+.4f}",
            f"max_dot={float(np.max(radial_dot)) if radial_dot.size else 0.0:+.4f}",
            flush=True,
        )

        # Configure C++ regular-grid field capture covering the full scene volume.
        # nx/ny/nz map to X(horizontal)/Y(top-view vertical)/Z(side-view vertical).
        # Keep grid small — it is a volumetric capture, not an image buffer.
        self._field_nx = 128
        self._field_ny = 64
        self._field_nz = 64
        if self.field_capture:
            bmin = np.array([self.scene.x_min, -self.scene.view_radius, -self.scene.view_radius], dtype=np.float32)
            bmax = np.array([self.scene.x_max,  self.scene.view_radius,  self.scene.view_radius], dtype=np.float32)
            self.tracer.enable_field_capture_regular(
                self._field_nx, self._field_ny, self._field_nz,
                bmin, bmax,
                capture_strikes=True,
                clear_existing=True,
            )
        else:
            print("[field-capture] disabled via --no-field", flush=True)

        self._build_uv_page_bank()
        n_lens_param = self._register_lens_parametric_groups()
        print(f"[parametric-register] lens_surface_groups={n_lens_param}", flush=True)
        self._register_uv_page_bank()

    def _uv_coords_for_tri_ids(self, tri_ids: np.ndarray, mode: str) -> np.ndarray:
        ids = np.ascontiguousarray(np.asarray(tri_ids, dtype=np.int32).reshape(-1), dtype=np.int32)
        verts = self.tri_vertices[ids] if ids.size else np.zeros((0, 3, 3), dtype=np.float64)
        uv = np.zeros((ids.size, 3, 2), dtype=np.float32)
        if ids.size <= 0:
            return uv
        if mode == "x_cylinder":
            y = verts[:, :, 1]
            z = verts[:, :, 2]
            x = verts[:, :, 0]
            ang = np.arctan2(z, y)
            uv[:, :, 0] = ((ang + math.pi) / (2.0 * math.pi)).astype(np.float32)
            xmin = float(np.min(x)); xmax = float(np.max(x)); span = max(EPS, xmax - xmin)
            uv[:, :, 1] = np.clip((x - xmin) / span, 0.0, 1.0).astype(np.float32)
        else:
            if mode == "xy":
                a = verts[:, :, 0]; b = verts[:, :, 1]
            elif mode == "xz":
                a = verts[:, :, 0]; b = verts[:, :, 2]
            else:
                a = verts[:, :, 1]; b = verts[:, :, 2]
            amin = float(np.min(a)); amax = float(np.max(a)); aspan = max(EPS, amax - amin)
            bmin = float(np.min(b)); bmax = float(np.max(b)); bspan = max(EPS, bmax - bmin)
            uv[:, :, 0] = np.clip((a - amin) / aspan, 0.0, 1.0).astype(np.float32)
            uv[:, :, 1] = np.clip((b - bmin) / bspan, 0.0, 1.0).astype(np.float32)
        return np.ascontiguousarray(uv, dtype=np.float32)

    def _build_uv_page_bank(self) -> None:
        bank = UvPageBank(self.n_bands, res=UV_PAGE_RES_DEFAULT, hot_limit=UV_HOT_GROUP_LIMIT_DEFAULT)
        bank.add("emitters", "emitter", self.source_tri_ids, self._uv_coords_for_tri_ids(self.source_tri_ids, "yz"))
        bank.add("object_plane", "object", self.object_tri_ids, self._uv_coords_for_tri_ids(self.object_tri_ids, "yz"))
        bank.add("sensor_plate", "sensor", self.image_plate_tri_ids, self._uv_coords_for_tri_ids(self.image_plate_tri_ids, "yz"))
        bank.add("aperture_or_iris", "aperture", self.aperture_stop_tri_ids, self._uv_coords_for_tri_ids(self.aperture_stop_tri_ids, "yz"))
        # Use only the cylindrical tube walls (silver + black) for UV display.
        # The camera_body triangles (flat annular caps + frustum cone) are excluded:
        # flat discs are degenerate under x_cylinder UV (constant V) and project
        # as full-height vertical lines in the screen view.
        bank.add("tube_or_baffles", "baffle", self.tube_wall_tri_ids, self._uv_coords_for_tri_ids(self.tube_wall_tri_ids, "x_cylinder"))
        bank.add("camera_barrel",    "baffle", self.camera_barrel_tri_ids,    self._uv_coords_for_tri_ids(self.camera_barrel_tri_ids,    "x_cylinder"))
        bank.add("camera_rear_cap",  "baffle", self.camera_rear_cap_tri_ids,  self._uv_coords_for_tri_ids(self.camera_rear_cap_tri_ids,  "yz"))
        bank.add("camera_front_cap", "baffle", self.camera_front_cap_tri_ids, self._uv_coords_for_tri_ids(self.camera_front_cap_tri_ids, "yz"))
        bank.add("camera_frustum",   "baffle", self.camera_frustum_tri_ids,   self._uv_coords_for_tri_ids(self.camera_frustum_tri_ids,   "x_cylinder"))
        bank.add("red_leak_probe",   "source", self.red_probe_tri_ids,         self._uv_coords_for_tri_ids(self.red_probe_tri_ids,         "yz"))
        # The red probe is a diagnostic emitter and must always be hot regardless
        # of layer budget; force it after add() so register_all() sees hot=True.
        for g in bank.groups:
            if g.name == "red_leak_probe":
                g.hot = True
        for i, (front_ids, back_ids, _rf, _rb) in enumerate(getattr(self, "lens_surface_groups", [])):
            bank.add(f"lens_{i:02d}_front", "lens_front", front_ids, self._uv_coords_for_tri_ids(front_ids, "yz"))
            bank.add(f"lens_{i:02d}_back", "lens_back", back_ids, self._uv_coords_for_tri_ids(back_ids, "yz"))
        self.uv_page_bank = bank
        for g in bank.groups:
            if g.name == "red_leak_probe":
                print(
                    "[red-probe-group]",
                    f"hot={int(g.hot)}",
                    f"layer={int(g.layer)}",
                    f"tris={int(g.tri_ids.size)}",
                    flush=True,
                )
        mem = bank.memory_report()
        print(
            "[uv-bank]",
            f"groups={int(mem['groups'])}",
            f"hot_limit={UV_HOT_GROUP_LIMIT_DEFAULT}",
            f"page={UV_PAGE_RES_DEFAULT}x{UV_PAGE_RES_DEFAULT}",
            f"channels={int(mem['channels'])}",
            f"gpu_hot_mb~={mem['gpu_hot_mb']:.1f}",
            f"warm_rgba32f_mb={mem['warm_rgba32f_mb']:.1f}",
            flush=True,
        )

    def _register_uv_page_bank(self) -> None:
        if self.uv_page_bank is None:
            return
        sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
        self.uv_page_bank.register_all(self.tracer, sample_area)
        print(
            "[uv-bank-register]",
            " ".join(f"{g.name}:gid={g.group_id}:layer={g.layer}:tris={g.tri_ids.size}" for g in self.uv_page_bank.groups),
            flush=True,
        )

    def _register_lens_parametric_groups(self) -> int:
        """Register passive exact lens-surface refiners for the live pipeline.

        The SDF-sphere formula computes a displacement in WORLD METRES:
            delta = k * (cu² + cv²),   k = 0.5 / radius_param
        where cu = u - 1/3, cv = v - 1/3 are barycentric offsets from the
        triangle centroid.  For this to equal the true spherical sag at the
        corner vertices (delta_vertex ≈ L²/(8·R_curv)) we need:
            k = 9·L²/(40·R_curv)   →   radius_param = 20·R_curv / (9·L²)
        where L is the mean edge length of the surface's triangles in metres.
        Passing the raw radius-of-curvature (metres) instead of radius_param
        gives displacements O(1/R) ~ metres, which is catastrophically wrong.
        """
        sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
        param_sdf_sphere = int(getattr(_sk, "TRI_PARAM_SURFACE_SDF_SPHERE", 3))
        param_none_role = 0
        n_lens_param = 0
        margin = 0.12  # barycentric UV neighbourhood radius for blended eval
        for front_ids, back_ids, r_front, r_back in getattr(self, "lens_surface_groups", []):
            for ids, r_curv in [(front_ids, r_front), (back_ids, r_back)]:
                if ids.size <= 0:
                    continue
                r_curv = abs(float(r_curv))
                if r_curv < 1.0e-6:
                    continue
                # Mean edge length of this surface's triangles (metres).
                verts = self.tri_vertices[ids]          # (n, 3, 3)
                e1 = verts[:, 1] - verts[:, 0]          # (n, 3)
                e2 = verts[:, 2] - verts[:, 0]
                e3 = verts[:, 2] - verts[:, 1]
                mean_L = float(np.mean([
                    np.mean(np.linalg.norm(e, axis=1))
                    for e in [e1, e2, e3]
                ]))
                mean_L = max(mean_L, 1.0e-8)
                # Scale radius_param so delta_vertex ≈ L²/(8·R_curv).
                radius_param = 20.0 * r_curv / (9.0 * mean_L ** 2)
                self.tracer.register_tri_group(
                    param_none_role,
                    sample_area,
                    np.ascontiguousarray(ids, dtype=np.int32),
                    parametric_surface={
                        "kind": param_sdf_sphere,
                        "coeffs": np.array([radius_param, margin], dtype=np.float64),
                    },
                )
                n_lens_param += 1
        return n_lens_param

    def _world_to_field_ijk(self, p: np.ndarray) -> Tuple[int, int, int]:
        x_span = max(EPS, float(self.scene.x_max - self.scene.x_min))
        yz_span = max(EPS, float(2.0 * self.scene.view_radius))
        ix = int(round(((float(p[0]) - self.scene.x_min) / x_span) * (self._field_nx - 1)))
        iy = int(round(((float(p[1]) + self.scene.view_radius) / yz_span) * (self._field_ny - 1)))
        iz = int(round(((float(p[2]) + self.scene.view_radius) / yz_span) * (self._field_nz - 1)))
        return ix, iy, iz

    def _configure_sensor_film_pipeline(self) -> None:
        db = SensorFilmDatabase.instance()
        plate = self.scene.image_plate
        sensor_w_mm = float(2.0 * plate.radius * 1000.0)
        sensor_h_mm = float(2.0 * plate.radius * 1000.0)
        pixel_pitch_um = float((sensor_w_mm / max(1, int(plate.pixels))) * 1000.0)
        focal_mm = float(max(1.0e-3, (plate.x - _scene_lenses(self.scene)[-1].center_x) * 1000.0))
        aperture_diam_mm = float(max(1.0e-3, 2.0 * _scene_lenses(self.scene)[0].aperture_radius * 1000.0))
        f_number = float(max(0.1, focal_mm / aperture_diam_mm))

        sensor_id = db.register_sensor(
            "thick_lens_focus_lab_sensor",
            {
                "focal_mm": focal_mm,
                "f_number": f_number,
                "aperture_diam_mm": aperture_diam_mm,
                "sensor_w_mm": sensor_w_mm,
                "sensor_h_mm": sensor_h_mm,
                "pixel_pitch_um": pixel_pitch_um,
                "qe_peak": 0.78,
                "full_well_e": 150_000.0,
                "read_noise_e": 2.5,
                "dark_current_e_s": 0.1,
            },
        )
        film_id = db.register_film(
            "thick_lens_focus_lab_film",
            {
                "iso": 100.0,
                "exposure_time_s": 0.010,
                "quantum_efficiency": 0.95,
                "target_grey_point": 0.18,
                "n_layers": 1,
            },
        )
        tensors = db.build_tensors()
        self._sensor_film_slots = [(int(sensor_id), int(film_id))] + [(-1, -1)] * (MAX_SENSOR_FILM_SLOTS - 1)
        self.tracer.set_sensor_film_ssbo(
            sensor_chunk=np.ascontiguousarray(tensors["sensor"], dtype=np.float32),
            film_chunk=np.ascontiguousarray(tensors["film"], dtype=np.float32),
            active_slots=self._sensor_film_slots,
        )

    def _ensure_bdpt_plate_sensor_group(self, n_px: int, n_aperture_samples: int, camera_mode: int = 2) -> int:
        cfg = (int(n_px), int(n_aperture_samples), int(camera_mode))
        if self._bdpt_sensor_cfg == cfg and self.bdpt_last_sensor_gid >= 0:
            return int(self.bdpt_last_sensor_gid)

        plate = self.scene.image_plate
        lenses = _scene_lenses(self.scene)
        first_lens = lenses[0]
        last_lens = lenses[-1]
        lens_stack_center_x = 0.5 * (first_lens.x_front + last_lens.x_back)
        stop_plane_x = float(last_lens.center_x)
        stop_radius_m = float(max(0.001, first_lens.aperture_radius))
        iris_cfg = getattr(self.scene, "iris_aperture", None)
        if iris_cfg is not None and bool(getattr(iris_cfg, "enabled", False)):
            stop_plane_x = float(iris_cfg.x_pos)
            stop_radius_m = float(max(1.0e-4, iris_cfg.r_inner))
        exit_pupil_radius = float(getattr(self.scene, "exit_pupil_radius", 0.0))
        if exit_pupil_radius > 0.0:
            stop_plane_x = float(getattr(self.scene, "exit_pupil_x", stop_plane_x))
            stop_radius_m = float(max(1.0e-4, exit_pupil_radius))

        role_emissive = int(getattr(_sk, "TRI_GROUP_ROLE_EMISSIVE", 1))
        role_sensor = int(getattr(_sk, "TRI_GROUP_ROLE_SENSOR", 2))
        role_blocker = int(getattr(_sk, "TRI_GROUP_ROLE_BLOCKER", 4))
        sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
        sample_pixel_cone = int(getattr(_sk, "TRI_GROUP_SAMPLE_PIXEL_CONE", 3))
        self.tracer.clear_tri_groups()
        self._bdpt_source_gid = int(
            self.tracer.register_tri_group(
                role_emissive,
                sample_area,
                np.ascontiguousarray(self.source_tri_ids, dtype=np.int32),
            )
        )
        self._bdpt_red_probe_gid = -1
        if int(self.red_probe_tri_ids.size) > 0:
            self._bdpt_red_probe_gid = int(
                self.tracer.register_tri_group(
                    role_emissive,
                    sample_area,
                    np.ascontiguousarray(self.red_probe_tri_ids, dtype=np.int32),
                )
            )
        aperture_stop_gid = -1
        if int(self.aperture_stop_tri_ids.size) > 0:
            aperture_stop_gid = int(
                self.tracer.register_tri_group(
                    role_blocker,
                    sample_area,
                    np.ascontiguousarray(self.aperture_stop_tri_ids, dtype=np.int32),
                )
            )
        has_aperture_geometry = aperture_stop_gid >= 0
        if not has_aperture_geometry:
            # No explicit aperture geometry: backend launches hemisphere rays.
            stop_radius_m = 0.0
        self.bdpt_last_sensor_gid = int(
            self.tracer.register_tri_group(
                role_sensor,
                sample_pixel_cone,
                np.ascontiguousarray(self.image_plate_tri_ids, dtype=np.int32),
                sensor_camera={
                    "pos": np.array([plate.x, 0.0, 0.0], dtype=np.float64),
                    "fwd": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                    "up": np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    "sensor_w_m": float(2.0 * plate.radius),
                    "sensor_h_m": float(2.0 * plate.radius),
                    # Kept for ABI compatibility; backend uses this as the
                    # explicit aperture radius (0 => hemisphere launch).
                    "focal_m": float(max(0.05, plate.x - stop_plane_x)),
                    "aperture_radius_m": float(stop_radius_m),
                    "n_px": int(n_px),
                    "n_py": int(n_px),
                    "n_aperture_samples": int(max(1, n_aperture_samples)),
                    "aperture_stop_group_id": int(aperture_stop_gid),
                    "effective_focal_m": float(max(1.0e-3, plate.x - lens_stack_center_x)),
                    "focus_distance_m": float(max(1.0e-3, lens_stack_center_x - self.scene.object_plane.x)),
                    "lens_center": np.array([lens_stack_center_x, 0.0, 0.0], dtype=np.float64),
                    "lens_fwd": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
                    "camera_mode": int(camera_mode),
                },
            )
        )
        # Re-register physical UV pages — clear_tri_groups() above destroyed them.
        # NOTE: _register_lens_parametric_groups() is intentionally NOT called here.
        # See the comment in __init__ for the full explanation.
        self._register_uv_page_bank()

        self._bdpt_sensor_cfg = cfg
        print(
            "[bdpt-register]",
            f"source_gid={self._bdpt_source_gid}",
            f"red_probe_gid={self._bdpt_red_probe_gid}",
            f"sensor_gid={self.bdpt_last_sensor_gid}",
            f"source_tris={int(self.source_tri_ids.size)}",
            f"red_probe_tris={int(self.red_probe_tri_ids.size)}",
            f"plate_tris={int(self.image_plate_tri_ids.size)}",
            f"stop_gid={int(aperture_stop_gid)}",
            f"stop_tris={int(self.aperture_stop_tri_ids.size)}",
            f"stop_x={stop_plane_x:.4f}",
            f"stop_r={stop_radius_m*1e3:.2f}mm",
            f"lens_center_x={lens_stack_center_x:.4f}",
            f"camera_mode={int(camera_mode)}",
            f"uv_groups={len(self.uv_page_bank.groups) if self.uv_page_bank is not None else 0}",
            flush=True,
        )
        return int(self.bdpt_last_sensor_gid)

    def reset_visual_integrators(self) -> None:
        with self._trace_lock:
            self._surface_top_ema = None
            self._surface_side_ema = None
            self._field_top_ema = None
            self._field_side_ema = None
            self._plate_rgb_accum = None
            self._plate_sensor_photons_accum = None
            self._plate_sensor_electrons_accum = None

    def clear_field_capture(self) -> None:
        with self._trace_lock:
            bmin = np.array([self.scene.x_min, -self.scene.view_radius, -self.scene.view_radius], dtype=np.float32)
            bmax = np.array([self.scene.x_max,  self.scene.view_radius,  self.scene.view_radius], dtype=np.float32)
            self.tracer.enable_field_capture_regular(
                int(self._field_nx),
                int(self._field_ny),
                int(self._field_nz),
                bmin,
                bmax,
                capture_strikes=True,
                clear_existing=True,
            )

    @staticmethod
    def _accumulate_image(previous: Optional[np.ndarray], current: np.ndarray, leak: float) -> np.ndarray:
        leak_f = float(np.clip(leak, 0.0, 1.0))
        cur = np.asarray(current, dtype=np.float32)
        if previous is None or previous.shape != cur.shape:
            return np.ascontiguousarray(cur, dtype=np.float32)
        return np.ascontiguousarray(leak_f * previous + cur, dtype=np.float32)

    def trace_forward(
        self,
        rays_per_emitter: int,
        seed: int,
        max_bounces: int = 6,
        decay: float = 0.97,
        tags: Optional[np.ndarray] = None,
    ) -> int:
        """Non-blocking forward trace via the persistent T1/T2/T3/T4 pipeline.

        Fans cosine-hemisphere rays from each source triangle and submits them
        to the machine immediately.  Returns the number of intents submitted.
        Output records arrive asynchronously via the background drain loop and
        are accumulated into tri_flux and _last_ray_segments.
        """
        rng = np.random.default_rng(seed)
        n_src  = int(self.src_pos.shape[0])
        n_rays = int(max(1, rays_per_emitter))
        total  = n_src * n_rays
        n_bands = self.n_bands

        origins     = np.empty((total, 3), dtype=np.float64)
        directions  = np.empty((total, 3), dtype=np.float64)
        amplitudes  = np.full((total, n_bands), complex(float(self.emitter_amp_gain)), dtype=np.complex128)
        src_ids     = np.empty((total,), dtype=np.int32)
        tag_arr     = np.zeros((total,), dtype=np.uint64)
        cflag_arr   = np.zeros((total,), dtype=np.uint8)  # 0 = emissive / forward

        for si in range(n_src):
            base   = si * n_rays
            normal = self.src_dir[si]
            up = np.array([0.0, 0.0, 1.0])
            if abs(normal[2]) > 0.9:
                up = np.array([1.0, 0.0, 0.0])
            tx = np.cross(normal, up);  tx /= np.linalg.norm(tx)
            ty = np.cross(normal, tx)

            u1 = rng.random(n_rays)
            u2 = rng.random(n_rays)
            cos_th = np.sqrt(1.0 - u1)
            sin_th = np.sqrt(u1)
            phi    = 2.0 * np.pi * u2
            dirs_local = (sin_th[:, None] * np.cos(phi)[:, None] * tx[None, :]
                        + sin_th[:, None] * np.sin(phi)[:, None] * ty[None, :]
                        + cos_th[:, None]                        * normal[None, :])
            dirs_local /= np.linalg.norm(dirs_local, axis=1, keepdims=True) + 1e-30

            origins   [base:base+n_rays] = self.src_pos[si]
            directions[base:base+n_rays] = dirs_local
            src_ids   [base:base+n_rays] = si
            if tags is not None and si < len(tags):
                tag_arr[base:base+n_rays] = int(tags[si])

        _use_gpu = self.compute_mode in ("gpu", "mixed")
        _all_gpu = self.compute_mode == "gpu"
        self.tracer.submit_rays(
            origins=np.ascontiguousarray(origins),
            directions=np.ascontiguousarray(directions),
            amplitudes=np.ascontiguousarray(amplitudes),
            src_ids=np.ascontiguousarray(src_ids),
            tags=np.ascontiguousarray(tag_arr) if tags is not None else None,
            color_flags=np.ascontiguousarray(cflag_arr),
            max_bounces=int(max_bounces),
            min_amplitude=float(self._min_amplitude),
            max_children=2,
            seed=int(seed),
            use_gpu_compute=_use_gpu,
            gpu_all_stages=_all_gpu,
            shader_dir=_SHADER_DIR,
        )

        self._ensure_drain_loop()
        return total

    def trace_sensor_cast(
        self,
        rays_per_sensor: int,
        seed: int,
        max_bounces: int = 6,
    ) -> int:
        """Back-cast rays from a pixel-UV grid on the sensor toward the aperture.

        Pixel sites are generated from a regular UV grid (sensor_res × sensor_res)
        clipped to the circular sensor disc.  The number of sites is always
        ≤ sensor_res², independent of the mesh tessellation density.

        Per pixel a small bokeh stencil is sampled: a random stencil centre is
        chosen on the aperture disc, then bokeh_rays points are drawn within
        bokeh_stencil_frac * aperture_radius of that centre.  This lets each
        frame probe a different aperture patch while keeping per-call ray counts
        small.
        """
        if self.aperture_radius <= 0.0 or self.image_plate_tri_ids.size == 0:
            return 0

        plate   = self.scene.image_plate
        res     = int(max(4, plate.sensor_res))
        n_rays  = max(1, int(rays_per_sensor) if rays_per_sensor > 0 else int(plate.bokeh_rays))
        s_frac  = float(plate.bokeh_stencil_frac)
        ap_r    = float(self.aperture_radius)
        stencil_r = s_frac * ap_r

        # ── Build circular pixel grid ────────────────────────────────────────
        # Regular UV cell centres in [-1, 1], clipped to unit disc.
        u   = np.linspace(-1.0, 1.0, res + 1)
        um  = 0.5 * (u[:-1] + u[1:])          # cell centres
        gy, gz = np.meshgrid(um, um, indexing='ij')  # (res, res)
        gy  = gy.ravel()
        gz  = gz.ravel()
        in_disc = (gy ** 2 + gz ** 2) <= 1.0
        gy  = gy[in_disc] * float(plate.radius)
        gz  = gz[in_disc] * float(plate.radius)
        n_pixels = int(gy.shape[0])
        if n_pixels == 0:
            return 0

        plate_x = float(plate.x)
        # Origins: pixel positions on the sensor face (all at plate_x, facing -X)
        sensor_cents = np.stack(
            [np.full(n_pixels, plate_x, dtype=np.float64), gy, gz], axis=1
        )  # (n_pixels, 3)

        n_sensor_channels = 3
        samples_per_pixel = n_rays * n_sensor_channels
        total = n_pixels * samples_per_pixel

        # ── Orthonormal basis for aperture disc ──────────────────────────────
        ap_n = self.aperture_normal / (np.linalg.norm(self.aperture_normal) + 1e-30)
        tb   = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ap_n, tb))) > 0.9:
            tb = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tb -= np.dot(tb, ap_n) * ap_n;  tb /= np.linalg.norm(tb) + 1e-30
        tc   = np.cross(ap_n, tb)

        rng  = np.random.default_rng(seed)

        # ── Bokeh stencil sampling ────────────────────────────────────────────
        # One random stencil centre per pixel (on full aperture disc).
        sc_ang = rng.uniform(0.0, 2.0 * math.pi, n_pixels)
        sc_rad = np.sqrt(rng.uniform(0.0, 1.0, n_pixels)) * ap_r
        stencil_cx = sc_rad * np.cos(sc_ang)   # (n_pixels,) in aperture-plane coords
        stencil_cy = sc_rad * np.sin(sc_ang)

        # n_rays RGB triplets per pixel within stencil_r of its centre.
        jit_ang = rng.uniform(0.0, 2.0 * math.pi, total)
        jit_rad = np.sqrt(rng.uniform(0.0, 1.0, total)) * stencil_r
        # Tile stencil centres across RGB triplets per pixel.
        s_cx = np.repeat(stencil_cx, samples_per_pixel) + jit_rad * np.cos(jit_ang)  # (total,)
        s_cy = np.repeat(stencil_cy, samples_per_pixel) + jit_rad * np.sin(jit_ang)
        # Clamp to aperture disc so stencil near the edge stays valid.
        s_rr = np.sqrt(s_cx ** 2 + s_cy ** 2)
        over = s_rr > ap_r
        if np.any(over):
            scale = ap_r / np.maximum(s_rr[over], 1e-30)
            s_cx[over] *= scale
            s_cy[over] *= scale

        ap_pts = (self.aperture_centroid
                  + s_cx[:, None] * tb[None, :]
                  + s_cy[:, None] * tc[None, :])  # (total, 3)

        # ── Assemble origins / directions ─────────────────────────────────────
        origins    = np.repeat(sensor_cents, samples_per_pixel, axis=0)   # (total, 3)
        d          = ap_pts - origins
        nrm        = np.linalg.norm(d, axis=1, keepdims=True)
        directions = d / np.maximum(nrm, 1e-30)
        src_ids    = np.repeat(np.arange(n_pixels, dtype=np.int32), samples_per_pixel)
        cflag_arr  = np.ones(total, dtype=np.uint8)   # 1 = sensor-cast / reverse

        if not self._sensor_aim_reported:
            center_origin = np.array([plate_x, 0.0, 0.0], dtype=np.float64)
            aim_vec = np.asarray(self.aperture_centroid, dtype=np.float64) - center_origin
            aim_len = float(np.linalg.norm(aim_vec))
            aim_dir = aim_vec / max(aim_len, 1.0e-30)
            print(
                "[sensor-aim]",
                f"plate_x={plate_x:.6f}",
                f"plate_radius={float(plate.radius):.6f}",
                f"target={np.asarray(self.aperture_centroid, dtype=np.float64).tolist()}",
                f"target_radius={float(ap_r):.6f}",
                f"center_dir={aim_dir.tolist()}",
                f"center_distance={aim_len:.6f}",
                f"pixels={n_pixels}",
                f"rays_per_pixel={n_rays}",
                flush=True,
            )
            self._sensor_aim_reported = True

        # Reverse paths are launched through RGB sensor sensitivity lobes.  This
        # makes the sensor an RGB spectral emitter instead of a flat white source.
        amp_scale = float(self.sensor_amp_gain)
        sens_rgb = _sensor_rgb_sensitivity_bands(self.freq_hz[:self.n_bands])
        channel_idx = np.tile(np.arange(n_sensor_channels, dtype=np.int32), n_pixels * n_rays)
        sensor_amps = (amp_scale * sens_rgb[channel_idx]).astype(np.complex128, copy=False)
        # Tag the channel in the high bits without disturbing the pixel src_id.
        tag_arr = (channel_idx.astype(np.uint64) << np.uint64(60)) | src_ids.astype(np.uint64)

        _use_gpu = self.compute_mode in ("gpu", "mixed")
        _all_gpu = self.compute_mode == "gpu"
        self.tracer.submit_rays(
            origins=np.ascontiguousarray(origins),
            directions=np.ascontiguousarray(directions),
            amplitudes=np.ascontiguousarray(sensor_amps),
            src_ids=np.ascontiguousarray(src_ids),
            tags=np.ascontiguousarray(tag_arr),
            color_flags=np.ascontiguousarray(cflag_arr),
            max_bounces=int(max_bounces),
            min_amplitude=float(self.sensor_min_amplitude),
            # Camera subpaths use one sampled continuation per surface event.
            # Deterministic two-way splitting from every sensor pixel explodes
            # as O(2^bounce) and will keep the GPU queue saturated at 256².
            max_children=1,
            seed=int(seed ^ 0xBEEF),
            use_gpu_compute=_use_gpu,
            gpu_all_stages=_all_gpu,
            shader_dir=_SHADER_DIR,
        )
        self._ensure_drain_loop()
        return total

    def _ensure_drain_loop(self) -> None:
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        # First pipeline start: configure adaptive thresholds
        # (pipeline now exists because submit_rays was just called above).
        if not self._pipeline_configured:
            self.tracer.set_min_amplitude(self._min_amplitude)
            self.tracer.precompute_epsilon_material_flags()
            self.tracer.set_intent_shuffle(float(self.intent_shuffle))
            self._pipeline_configured = True
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="pipeline-drain")
        self._drain_thread.start()

    def _drain_loop(self) -> None:
        import time
        _min_batch =   256
        _max_batch = 32_768
        # Throughput tracking for adaptive batch sizing.
        # The target drain window is an alpha-weighted mix of:
        #   - a base 20 ms constant (pipeline-driven)
        #   - a fraction (50%) of the per-frame budget at the preferred FPS
        # fps_alpha steers how strongly the frame-rate preference matters.
        _BASE_DRAIN_S    = 0.020
        _FRAME_FRACTION  = 0.50   # drain may use up to this share of one frame
        _throughput_est  = 4_000.0  # initial guess: 4k records/sec
        _ema_alpha       = 0.25     # EMA weight for throughput update
        _t_last          = time.perf_counter()
        _t_profile_last  = time.perf_counter()  # for periodic stats snapshots
        _n_since_update  = 0
        _PROFILE_INTERVAL_S  = 0.5   # snapshot every 500 ms
        _PROFILE_WARMUP      = 8     # print calibration report after N non-trivial snapshots
        while not self._drain_stop.is_set():
            try:
                # Blend frame-rate preference into the target drain window.
                fps         = max(1.0, float(self.target_fps))
                alpha       = float(self.fps_alpha)
                fps_budget  = _FRAME_FRACTION / fps
                target_s    = (1.0 - alpha) * _BASE_DRAIN_S + alpha * fps_budget

                # fps budget is the CEILING on work per drain cycle.
                # If the queue is deep we still respect it — backpressure is
                # the right relief valve, not burning through all of in_flight.
                in_flight  = int(self.tracer.in_flight_count())
                batch_tp   = max(_min_batch, min(_max_batch, int(_throughput_est * target_s)))
                # Allow a small queue-depth boost only up to the fps ceiling,
                # so a temporarily deep queue doesn't blow the frame budget.
                batch_q    = max(_min_batch, min(batch_tp, in_flight * 4))
                batch      = max(batch_tp, batch_q)
                records    = self.tracer.drain_records_slim(max_n=batch)
                n          = int(records["kind"].shape[0]) if records else 0
                if n > 0:
                    self._accumulate_records(records)
                    # Update rolling throughput estimate every 50ms or 5k records.
                    _n_since_update += n
                    t_now = time.perf_counter()
                    dt    = t_now - _t_last
                    if dt >= 0.050 or _n_since_update >= 5_000:
                        _throughput_est = (
                            (1.0 - _ema_alpha) * _throughput_est +
                            _ema_alpha * (_n_since_update / max(dt, 1e-9))
                        )
                        _t_last         = t_now
                        _n_since_update = 0

                    # ── Live GPU profiling ───────────────────────────────────
                    t_now2 = time.perf_counter()
                    if not self._gpu_calibrated and (t_now2 - _t_profile_last) >= _PROFILE_INTERVAL_S:
                        _t_profile_last = t_now2
                        try:
                            s = self.tracer.pipeline_stats()
                        except Exception:
                            s = None
                        if s:
                            # Count as non-trivial if any GPU work has been done.
                            gpu_total = sum(
                                s[stage].get("gpu_processed", 0)
                                for stage in ("t1", "t2", "t3", "t4")
                            )
                            if gpu_total > 0:
                                self._gpu_profile_snapshots.append(s)
                            if len(self._gpu_profile_snapshots) >= _PROFILE_WARMUP:
                                self._print_gpu_calibration_report()
                                self._gpu_calibrated = True
                    # If we filled the batch, the queue has more — loop immediately.
                    if n >= batch * 3 // 4:
                        continue
                else:
                    # Back off harder when the pipeline is also empty — avoids
                    # spinning the CPU core at full speed in GPU-all mode.
                    if int(self.tracer.in_flight_count()) == 0:
                        time.sleep(0.020)
                    else:
                        time.sleep(0.001)
            except Exception as _drain_exc:
                print(f"[drain-loop ERROR] {_drain_exc}", flush=True)
                time.sleep(0.1)

    def _print_gpu_calibration_report(self) -> None:
        """Print a live calibration report from accumulated pipeline stat snapshots."""
        snaps = self._gpu_profile_snapshots
        if not snaps:
            return
        stages = ("t1", "t2", "t3", "t4")
        print("[gpu-calibration] === pipeline efficiency report ===", flush=True)
        for stage in stages:
            cpu_tp  = float(np.mean([s[stage]["throughput"]     for s in snaps]))
            gpu_tp  = float(np.mean([s[stage]["gpu_throughput"] for s in snaps]))
            cpu_n   = int(snaps[-1][stage]["processed"])
            gpu_n   = int(snaps[-1][stage]["gpu_processed"])
            bs_cpu  = int(snaps[-1][stage]["batch_size"])
            bs_gpu  = int(snaps[-1][stage]["gpu_batch_size"])
            frac    = float(snaps[-1][stage]["gpu_fraction"])
            total   = cpu_tp + gpu_tp
            pct     = 100.0 * gpu_tp / total if total > 0 else 0.0
            print(
                f"[gpu-calibration]  {stage.upper()}:"
                f"  cpu={cpu_tp/1e3:7.1f}k/s (n={cpu_n:,}, bs={bs_cpu})"
                f"  gpu={gpu_tp/1e3:7.1f}k/s (n={gpu_n:,}, bs={bs_gpu})"
                f"  gpu_frac={frac:.2f} ({pct:.1f}%)",
                flush=True,
            )
        print("[gpu-calibration] =================================", flush=True)

    def _accumulate_records(self, records: dict) -> None:
        kinds     = records["kind"]        # uint8  (N,)
        bounces   = records["bounce"]      # int32  (N,)
        amp_re    = records["amp_re"]      # float32 (N, n_bands)
        amp_im    = records["amp_im"]      # float32 (N, n_bands)
        seg_start = records["seg_start"]   # float32 (N, 3)
        pos       = records["pos"]         # float32 (N, 3)
        cflags    = records["color_flag"]  # uint8  (N,)  0=emissive 1=sensor

        # Display class encoding:
        #   STRIKE + flag 0 → class 0 (forward strike)   FIELD + flag 0 → class 1 (forward volume)
        #   FIELD  + flag 1 → class 2 (reverse volume)   STRIKE + flag 1 → class 3 (reverse strike)
        strike_mask = kinds == 0   # RayRecordKind::STRIKE
        field_mask  = kinds == 3   # RayRecordKind::FIELD
        vis_mask    = strike_mask | field_mask
        if not np.any(vis_mask):
            return

        # --- Build unified segment rows (13 cols) ---
        # col 0:2  seg_start xyz
        # col 3:5  hit/field pos xyz
        # col 6    src_id
        # col 7    bounce
        # col 8    band 0 placeholder
        # col 9    band 0 amplitude |amp|
        # col 10   band 0 phase
        # col 11   path_at_seg_start
        # col 12   display_class (0-3)

        vm_re  = amp_re[vis_mask]
        vm_im  = amp_im[vis_mask]
        nb     = min(vm_re.shape[1], self.n_bands)
        amp_sq = vm_re[:, :nb] ** 2 + vm_im[:, :nb] ** 2

        vm_kinds  = kinds[vis_mask]
        vm_cflags = cflags[vis_mask].astype(np.int32)
        # class = is_field * 1 + color_flag * (is_strike ? 3 : 1)
        # Simpler: 0,1,2,3 mapped directly:
        #   strike+0 → 0   strike+1 → 3   field+0 → 1   field+1 → 2
        is_field   = (vm_kinds == 3).astype(np.int32)
        disp_class = np.where(is_field,
                              1 + vm_cflags,        # field+0→1, field+1→2
                              vm_cflags * 3)        # strike+0→0, strike+1→3

        vm_ss  = seg_start[vis_mask]
        vm_pos = pos[vis_mask]
        # FIELD records: use pos as both endpoints (single dot in 3D space)
        actual_ss = np.where(is_field[:, None].astype(bool), vm_pos, vm_ss)

        # ── Accumulate hit positions directly into the fixed voxel display grid ──────
        # No ring-buffer; no per-row heap allocations; accumulation runs forever.
        amp_v = (np.sqrt(np.clip(amp_sq[:, 0], 0.0, None)) if nb > 0
                 else np.ones(int(np.count_nonzero(vis_mask)), dtype=np.float32)).astype(np.float32)

        # For class-3 also count the ray origin (image-plate surface) as a hit.
        rev_strike_mask = (disp_class == 3)
        if np.any(rev_strike_mask):
            vm_pos_all   = np.concatenate([vm_pos,   vm_ss[rev_strike_mask]],    axis=0)
            amp_v_all    = np.concatenate([amp_v,    amp_v[rev_strike_mask]],    axis=0)
            disp_cls_all = np.concatenate([disp_class, np.full(int(np.count_nonzero(rev_strike_mask)), 3, dtype=np.int32)], axis=0)
        else:
            vm_pos_all   = vm_pos
            amp_v_all    = amp_v
            disp_cls_all = disp_class

        # Pack raw hit positions into ring-buffer rows: (x, y, z, amp, class)
        pts = np.empty((vm_pos_all.shape[0], 5), dtype=np.float32)
        pts[:, :3] = vm_pos_all
        pts[:, 3]  = amp_v_all
        pts[:, 4]  = disp_cls_all.astype(np.float32)
        n   = pts.shape[0]
        cap = self._VIS_CAP
        with self._segs_lock:
            ptr = self._vis_ptr
            end = ptr + n
            if end <= cap:
                self._vis_buf[ptr:end] = pts
            else:
                first = cap - ptr
                self._vis_buf[ptr:] = pts[:first]
                # Write remaining rows from the start, wrapping once.
                # If the batch exceeds the ring capacity, keep only the tail.
                rem = n - first
                if rem >= cap:
                    self._vis_buf[:] = pts[n - cap:]
                    rem = cap
                else:
                    self._vis_buf[:rem] = pts[first:]
                self._vis_full = True
            self._vis_ptr = end % cap
            if end >= cap:
                self._vis_full = True

    def get_ray_segments(self) -> Optional[np.ndarray]:
        """Superseded by the voxel display grid (_disp_grid). Returns None."""
        return None

    def get_bdpt_records(self) -> Optional[np.ndarray]:
        """Return last captured backend BDPT endpoint records (N, 16)."""
        with self._trace_lock:
            return self._last_bdpt_records

    def build_all_prospective_reverse_segments(
        self,
        seed: int,
        max_segments: int = 300_000,
    ) -> np.ndarray:
        """Build reverse-ray overlay segments for every launched PIXEL_CONE sample.

        This is independent of BDPT record survival: every prospective reverse
        launch is represented as a visible line segment.
        """
        with self._trace_lock:
            n_px = int(max(1, self.bdpt_last_n_px))
            n_py = int(max(1, self.bdpt_last_n_px))
            n_ap = int(max(1, self.bdpt_last_aperture_samples))

            plate = self.scene.image_plate
            lenses = _scene_lenses(self.scene)
            first_lens = lenses[0]
            last_lens = lenses[-1]
            stop_plane_x = float(last_lens.center_x)
            stop_radius_m = float(max(0.001, first_lens.aperture_radius))
            iris_cfg = getattr(self.scene, "iris_aperture", None)
            if iris_cfg is not None and bool(getattr(iris_cfg, "enabled", False)):
                stop_plane_x = float(iris_cfg.x_pos)
                stop_radius_m = float(max(1.0e-4, iris_cfg.r_inner))
            exit_pupil_radius = float(getattr(self.scene, "exit_pupil_radius", 0.0))
            if exit_pupil_radius > 0.0:
                stop_plane_x = float(getattr(self.scene, "exit_pupil_x", stop_plane_x))
                stop_radius_m = float(max(1.0e-4, exit_pupil_radius))

            sensor_pos = np.array([float(plate.x), 0.0, 0.0], dtype=np.float64)
            fwd = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(fwd, up)
            right /= max(float(np.linalg.norm(right)), EPS)
            up /= max(float(np.linalg.norm(up)), EPS)

            sensor_w = float(2.0 * plate.radius)
            sensor_h = float(2.0 * plate.radius)
            focal_m = float(max(0.05, plate.x - stop_plane_x))
            aperture_centre = sensor_pos + fwd * focal_m
            pix_w = sensor_w / float(max(1, n_px))
            pix_h = sensor_h / float(max(1, n_py))

            total_rays = int(max(1, n_px * n_py * n_ap))
            keep = int(min(max_segments, total_rays))

            py_grid, px_grid = np.indices((n_py, n_px), dtype=np.float64)
            pix_pts = (
                sensor_pos[None, None, :]
                + (px_grid[:, :, None] + 0.5) * pix_w * right[None, None, :]
                + (py_grid[:, :, None] + 0.5) * pix_h * up[None, None, :]
            )

            pix_pts = np.repeat(pix_pts[:, :, None, :], n_ap, axis=2)
            ai = np.broadcast_to(np.arange(n_ap, dtype=np.float64)[None, None, :], (n_py, n_px, n_ap))

            rng = np.random.Generator(np.random.PCG64(int(seed) & ((1 << 63) - 1)))
            j1 = rng.random((n_py, n_px, n_ap), dtype=np.float64)
            j2 = rng.random((n_py, n_px, n_ap), dtype=np.float64)

            q = (ai + j1) / float(max(1, n_ap))
            r = np.sqrt(np.clip(q, 0.0, 1.0)) * float(stop_radius_m)
            golden = 2.39996322972865332
            th = golden * ai + 2.0 * math.pi * j2

            ap_pts = (
                aperture_centre[None, None, None, :]
                + r[..., None] * np.cos(th)[..., None] * right[None, None, None, :]
                + r[..., None] * np.sin(th)[..., None] * up[None, None, None, :]
            )

            d = ap_pts - pix_pts
            nd = np.linalg.norm(d, axis=3)
            valid = nd > EPS
            if not np.any(valid):
                return np.zeros((0, 2, 3), dtype=np.float32)

            du = np.zeros_like(d)
            du[valid] = d[valid] / nd[valid, None]

            x_span = float(self.scene.x_max - self.scene.x_min)
            target_x = np.where(du[..., 0] < 0.0, float(self.scene.x_min), float(self.scene.x_max))
            safe_dx = np.where(np.abs(du[..., 0]) < 1.0e-9, np.sign(du[..., 0]) * 1.0e-9 + 1.0e-9, du[..., 0])
            t_end = (target_x - ap_pts[..., 0]) / safe_dx
            t_end = np.where(t_end <= 1.0e-6, x_span, t_end)
            end_pts = ap_pts + du * t_end[..., None]

            p0 = pix_pts.reshape(-1, 3)
            p1 = end_pts.reshape(-1, 3)
            valid_flat = valid.reshape(-1)
            p0 = p0[valid_flat]
            p1 = p1[valid_flat]
            if p0.shape[0] <= 0:
                return np.zeros((0, 2, 3), dtype=np.float32)

            if p0.shape[0] > keep:
                idx = np.linspace(0, p0.shape[0] - 1, keep, dtype=np.int64)
                p0 = p0[idx]
                p1 = p1[idx]

            segs = np.stack([p0, p1], axis=1)
            return np.ascontiguousarray(segs.astype(np.float32, copy=False), dtype=np.float32)

    def capture_plate_rgb(self, pixels: int | None = None, leak: float = 1.0) -> np.ndarray:
        with self._trace_lock:
            n = int(max(4, pixels or self.scene.image_plate.pixels))
            plate = self.scene.image_plate
            r = float(plate.radius)
            ids = self.image_plate_tri_ids
            if ids.size <= 0:
                return np.zeros((n, n, 3), dtype=np.float32)
            verts = self.tri_vertices[ids]
            uv = np.empty((ids.size, 3, 2), dtype=np.float32)
            uv[:, :, 0] = np.clip((verts[:, :, 2] + r) / max(EPS, 2.0 * r), 0.0, 1.0)
            uv[:, :, 1] = np.clip((verts[:, :, 1] + r) / max(EPS, 2.0 * r), 0.0, 1.0)
            rgb = self.tracer.rasterize_tri_flux_uv(
                np.ascontiguousarray(self.tri_flux, dtype=np.float32),
                np.ascontiguousarray(ids, dtype=np.int32),
                np.ascontiguousarray(uv, dtype=np.float32),
                n,
                n,
                1.0,
                100.0,
            )["rgb_tonemapped"]
            self._plate_rgb_accum = self._accumulate_image(self._plate_rgb_accum, np.asarray(rgb, dtype=np.float32), leak)
            return np.clip(self._plate_rgb_accum, 0.0, 1.0).astype(np.float32, copy=False)

    def ideal_object_rgb(self, pixels: int | None = None) -> np.ndarray:
        n = int(max(4, pixels or self.scene.image_plate.pixels))
        # Omniscient pinhole ideal: forward integration in a copy of the scene
        # with optics disabled, camera placed at image-plate looking upstream.
        scene_no_optics = SceneConfig(
            x_min=self.scene.x_min,
            x_max=self.scene.x_max,
            source_x=self.scene.source_x,
            source_radius=self.scene.source_radius,
            tube_x0=self.scene.tube_x0,
            tube_x1=self.scene.tube_x1,
            tube_radius=self.scene.tube_radius,
            baffle0_x=self.scene.baffle0_x,
            baffle0_aperture=self.scene.baffle0_aperture,
            baffle1_x=self.scene.baffle1_x,
            baffle1_aperture=self.scene.baffle1_aperture,
            baffle2_x=self.scene.baffle2_x,
            baffle2_aperture=self.scene.baffle2_aperture,
            screen_x=self.scene.screen_x,
            screen_radius=self.scene.screen_radius,
            view_radius=self.scene.view_radius,
            enable_debug_plate=self.scene.enable_debug_plate,
            debug_plate_x=self.scene.debug_plate_x,
            debug_plate_radius=self.scene.debug_plate_radius,
            object_plane=self.scene.object_plane,
            image_plate=ImagePlateConfig(x=self.scene.image_plate.x, radius=self.scene.image_plate.radius, pixels=n),
            side_room_y=self.scene.side_room_y,
            side_room_aperture=self.scene.side_room_aperture,
            side_room_bore_radius=self.scene.side_room_bore_radius,
            side_room_source_radius=self.scene.side_room_source_radius,
            side_room_wall_outer=self.scene.side_room_wall_outer,
            side_room_depth=self.scene.side_room_depth,
            side_room_source_offset=self.scene.side_room_source_offset,
            side_room_source_emission=self.scene.side_room_source_emission,
            side_room_diffuser_enabled=self.scene.side_room_diffuser_enabled,
            side_room_diffuser_radius=self.scene.side_room_diffuser_radius,
            side_room_diffuser_thickness=self.scene.side_room_diffuser_thickness,
            side_room_diffuser_transmittance=self.scene.side_room_diffuser_transmittance,
            side_room_diffuser_diffuse_frac=self.scene.side_room_diffuser_diffuse_frac,
            side_room_diffuser_tilt_x_deg=self.scene.side_room_diffuser_tilt_x_deg,
            side_room_diffuser_tilt_z_deg=self.scene.side_room_diffuser_tilt_z_deg,
            side_room_source_x=self.scene.side_room_source_x,
            side_room_source_z=self.scene.side_room_source_z,
            stage_probe_emitter_enabled=self.scene.stage_probe_emitter_enabled,
            stage_probe_x=self.scene.stage_probe_x,
            stage_probe_y=self.scene.stage_probe_y,
            stage_probe_z=self.scene.stage_probe_z,
            stage_probe_radius=self.scene.stage_probe_radius,
            disable_optics=True,
            lens=self.scene.lens,
            lens_stack=list(self.scene.lens_stack),
        )
        ideal_bench = ForwardCppLensBench(
            scene=scene_no_optics,
            freq_hz=self.freq_hz.copy(),
            view_h=max(16, n),
            view_w=max(16, n),
            sidecar=self.sidecar,
        )
        spec = np.zeros((ideal_bench.n_bands, n, n), dtype=np.float32)
        plate = scene_no_optics.image_plate
        fov_rad = 2.0 * math.atan2(float(plate.radius), max(EPS, float(plate.x - scene_no_optics.object_plane.x)))
        ideal_bench.tracer.integrate_image_into(
            src_pos=ideal_bench.src_pos,
            src_dir=ideal_bench.src_dir,
            src_directivity=ideal_bench.src_directivity,
            cam_pos=np.array([plate.x, 0.0, 0.0], dtype=np.float64),
            cam_fwd=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            cam_up=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            out_image=spec,
            fov_rad=float(max(0.15, min(math.pi - 0.1, fov_rad))),
            n_rays=24,
            max_bounces=16,
            min_amplitude=1.0e-7,
            seed=20260512,
        )
        return _spectral_image_to_rgb(spec, ideal_bench.freq_hz)

    def capture_plate_bdpt_rgb(
        self,
        pixels: int | None = None,
        aperture_samples: int = 2,
        seed: int = 0,
        max_records: int = 300_000,
        max_bounces: int | None = None,
        leak: float = 1.0,
        n_rays_bdpt: int | None = None,
        camera_mode: int = 2,
    ) -> np.ndarray:
        with self._trace_lock:
            n = int(max(4, pixels or self.scene.image_plate.pixels))
            aperture_samples_i = int(max(1, aperture_samples))
            self.bdpt_last_n_px = int(n)
            self.bdpt_last_aperture_samples = int(aperture_samples_i)
            sensor_gid = self._ensure_bdpt_plate_sensor_group(
                n,
                aperture_samples_i,
                camera_mode=int(camera_mode),
            )
            bounce_cap = int(max_bounces) if max_bounces is not None else 64
            # BDPT launch count must not be tied to aperture sample count alone.
            # Keep a substantive per-frame budget so reverse paths actually form.
            if n_rays_bdpt is None:
                n_rays_bdpt = int(max(64, min(8192, 8 * n)))
            else:
                n_rays_bdpt = int(max(1, n_rays_bdpt))
            # Size output records to backend worst-case for this batch so
            # backward sensor paths are not clipped by an undersized buffer.
            light_path_budget = int(n_rays_bdpt) * int(max(1, bounce_cap) + 1)
            recs_per_sample = int(max(1, bounce_cap) + 2 + int(self.n_bands))
            sensor_path_budget = int(n) * int(n) * int(aperture_samples_i) * int(recs_per_sample)
            max_records_use = int(max(max_records, light_path_budget + sensor_path_budget))
            self.bdpt_last_launched_rays = int(n_rays_bdpt)
            records = self.tracer.bidirectional(
                int(max(1, n_rays_bdpt)),
                int(max(1, bounce_cap)),
                1.0e-7,
                int(seed),
                int(max_records_use),
            )
            rec_arr = np.asarray(records, dtype=np.float32)
            if rec_arr.ndim == 2 and rec_arr.shape[1] >= 16 and rec_arr.shape[0] > 0:
                # Keep contiguous ordering so per-subpath sorting/reconstruction
                # can build visible exploratory ray segments.
                keep_n = min(int(rec_arr.shape[0]), 220_000)
                self._last_bdpt_records = np.ascontiguousarray(rec_arr[:keep_n, :16], dtype=np.float32)
            else:
                self._last_bdpt_records = None
            self.bdpt_last_records = int(np.asarray(records).shape[0]) if np.asarray(records).ndim == 2 else 0
            self.bdpt_last_volume_records = 0
            self.bdpt_last_volume_power = 0.0
            try:
                vol = self.tracer.accumulate_endpoint_records_to_field_capture(
                    records,
                    int(sensor_gid),
                    True,
                    True,
                )
                if isinstance(vol, dict):
                    self.bdpt_last_volume_records = int(vol.get("written_records", 0))
                    self.bdpt_last_volume_power = float(vol.get("written_power", 0.0))
                    self.bdpt_cumulative_volume_records += int(self.bdpt_last_volume_records)
                    self.bdpt_cumulative_volume_power += float(self.bdpt_last_volume_power)
            except Exception:
                self.bdpt_last_volume_records = 0
                self.bdpt_last_volume_power = 0.0
            reduced = self.tracer.reduce_endpoint_records_to_rgb_image(
                records,
                n,
                n,
                int(sensor_gid),
                1.0,
                100.0,
            )
            rgb_tm = reduced.get("rgb_tonemapped", None) if isinstance(reduced, dict) else None
            telemetry = reduced.get("telemetry", {}) if isinstance(reduced, dict) else {}
            if isinstance(telemetry, dict):
                self.bdpt_last_telemetry = {
                    str(k): int(v) for k, v in telemetry.items()
                    if isinstance(v, (int, np.integer))
                }
                self.bdpt_last_survivor_records = int(self.bdpt_last_telemetry.get("kept_records", 0))
                if self.bdpt_last_survivor_records <= 0:
                    self.bdpt_consecutive_no_survivor_frames += 1
                else:
                    self.bdpt_consecutive_no_survivor_frames = 0
            else:
                self.bdpt_last_telemetry = {}
                self.bdpt_last_survivor_records = 0
                self.bdpt_consecutive_no_survivor_frames += 1

            sensor_integral = self.tracer.reduce_endpoint_records_to_sensor_integral(
                records,
                n,
                n,
                int(sensor_gid),
                0.0,
                1.0,
            )
            metrics = sensor_integral.get("metrics", {}) if isinstance(sensor_integral, dict) else {}
            if isinstance(metrics, dict):
                self.bdpt_last_telemetry.update({
                    str(k): int(v) for k, v in metrics.items()
                    if isinstance(v, (int, np.integer))
                })

            photons = sensor_integral.get("photons_per_pixel", None) if isinstance(sensor_integral, dict) else None
            electrons = sensor_integral.get("electrons_per_pixel", None) if isinstance(sensor_integral, dict) else None
            if photons is not None and electrons is not None:
                photons_arr = np.asarray(photons, dtype=np.float32)
                electrons_arr = np.asarray(electrons, dtype=np.float32)
                if photons_arr.shape == (n, n) and electrons_arr.shape == (n, n):
                    self.bdpt_last_sensor_photons = float(np.sum(np.asarray(photons_arr, dtype=np.float64)))
                    self.bdpt_last_sensor_power = float(np.sum(np.asarray(electrons_arr, dtype=np.float64)))
                else:
                    self.bdpt_last_sensor_photons = 0.0
                    self.bdpt_last_sensor_power = 0.0
            else:
                self.bdpt_last_sensor_photons = 0.0
                self.bdpt_last_sensor_power = 0.0

            if rgb_tm is None:
                self.bdpt_debug_print_counter += 1
                print(
                    "[bdpt-debug]",
                    f"call={self.bdpt_debug_print_counter}",
                    f"launched={self.bdpt_last_launched_rays}",
                    f"endpoint_records={self.bdpt_last_records}",
                    f"kept={self.bdpt_last_survivor_records}",
                    f"sensor_gid={sensor_gid}",
                    f"vol_records={self.bdpt_last_volume_records}",
                    f"photons={self.bdpt_last_sensor_photons:.3e}",
                    f"sensor_power={self.bdpt_last_sensor_power:.3e}",
                    "rgb_tonemapped=NONE",
                    flush=True,
                )
                return np.zeros((n, n, 3), dtype=np.float32)
            rgb_tm_arr = np.asarray(rgb_tm, dtype=np.float32)
            if rgb_tm_arr.shape != (n, n, 3):
                self.bdpt_debug_print_counter += 1
                print(
                    "[bdpt-debug]",
                    f"call={self.bdpt_debug_print_counter}",
                    f"launched={self.bdpt_last_launched_rays}",
                    f"endpoint_records={self.bdpt_last_records}",
                    f"kept={self.bdpt_last_survivor_records}",
                    f"sensor_gid={sensor_gid}",
                    f"vol_records={self.bdpt_last_volume_records}",
                    f"photons={self.bdpt_last_sensor_photons:.3e}",
                    f"sensor_power={self.bdpt_last_sensor_power:.3e}",
                    f"rgb_shape={tuple(rgb_tm_arr.shape)}",
                    flush=True,
                )
                return np.zeros((n, n, 3), dtype=np.float32)
            lit_mask = np.sum(np.asarray(rgb_tm_arr, dtype=np.float64), axis=2) > 1.0e-8
            lit_pixels = int(np.count_nonzero(lit_mask))
            lit_fraction = float(lit_pixels) / float(max(1, n * n))
            self.bdpt_debug_print_counter += 1
            print(
                "[bdpt-debug]",
                f"call={self.bdpt_debug_print_counter}",
                f"launched={self.bdpt_last_launched_rays}",
                f"endpoint_records={self.bdpt_last_records}",
                f"sensor_group_records={self.bdpt_last_telemetry.get('sensor_group_records', 0)}",
                f"kept={self.bdpt_last_survivor_records}",
                f"kept_pixel={self.bdpt_last_telemetry.get('kept_pixel_cone_records', 0)}",
                f"kept_projected={self.bdpt_last_telemetry.get('kept_projected_records', 0)}",
                f"wrong_group={self.bdpt_last_telemetry.get('drop_wrong_group', 0)}",
                f"non_pixel={self.bdpt_last_telemetry.get('drop_non_pixel_cone', 0)}",
                f"projection_failed={self.bdpt_last_telemetry.get('drop_projection_failed', 0)}",
                f"lit_pixels={lit_pixels}",
                f"lit_frac={lit_fraction:.4f}",
                f"sensor_gid={sensor_gid}",
                f"vol_records={self.bdpt_last_volume_records}",
                f"photons={self.bdpt_last_sensor_photons:.3e}",
                f"sensor_power={self.bdpt_last_sensor_power:.3e}",
                flush=True,
            )
            return np.clip(rgb_tm_arr, 0.0, 1.0).astype(np.float32, copy=False)

    def export_bdpt_ray_visualization(self, output_file: str = "bdpt_rays.txt") -> Dict[str, any]:
        """Export BDPT ray paths to a text file for detailed visualization.
        
        Creates a comprehensive breakdown of all backward rays from the last BDPT call,
        including bounce points, materials hit, path lengths, and amplitude values.
        This allows you to see exactly where each ray travels and why it terminates.
        
        Args:
            output_file: Output text file path (default: bdpt_rays.txt in current directory)
        
        Returns:
            Statistics dict with ray_count, endpoint_count, path length stats, and bounce distribution
        """
        with self._trace_lock:
            if self._last_bdpt_records is None:
                print(f"[ray-export] No BDPT records available yet")
                return {
                    "ray_count": 0,
                    "endpoint_count": 0,
                    "total_path_length": 0.0,
                    "error": "No BDPT records",
                }
            
            # Call the visualization export function
            stats = visualize_bdpt_rays_3d_export(
                self._last_bdpt_records,
                self.scene,
                output_file=output_file,
                max_rays=1000,
            )
            
            print(f"[ray-export] Exported {stats['ray_count']} rays to {output_file}")
            print(f"[ray-export] Total endpoints: {stats['endpoint_count']}")
            print(f"[ray-export] Avg path length: {stats['total_path_length'] / max(1, stats['ray_count']):.6f}m")
            print(f"[ray-export] Max path length: {stats['max_path_length']:.6f}m")
            print(f"[ray-export] Bounce distribution: {stats['bounce_distribution']}")
            
            return stats

    def orthographic_linear_views(
        self,
        field_gain: float = 0.25,
        surface_gain: float = 0.35,
        field_leak: float = 0.0,
        surface_leak: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._trace_lock:
            tri_flux_vis = np.ascontiguousarray(self.tri_flux, dtype=np.float32)
            if self._surface_suppress_tri_ids.size > 0:
                tri_flux_vis[self._surface_suppress_tri_ids, :] = 0.0

            # Render channels separately so each can be leaky-integrated with
            # independent time constants.
            s_top_rgb, s_side_rgb = self.tracer.compose_lens_bench_views(
                tri_flux=tri_flux_vis,
                tri_centroids=np.ascontiguousarray(self.tri_centroids, dtype=np.float64),
                view_h=int(self.view_h),
                view_w=int(self.view_w),
                x_min=float(self.scene.x_min),
                x_max=float(self.scene.x_max),
                view_radius=float(self.scene.view_radius),
                field_nx=int(self._field_nx),
                field_ny=int(self._field_ny),
                field_nz=int(self._field_nz),
                field_gain=0.0,
                surface_gain=float(surface_gain),
                hdr_white_percentile=99.8,
            )
            f_top_rgb, f_side_rgb = self.tracer.compose_lens_bench_views(
                tri_flux=tri_flux_vis,
                tri_centroids=np.ascontiguousarray(self.tri_centroids, dtype=np.float64),
                view_h=int(self.view_h),
                view_w=int(self.view_w),
                x_min=float(self.scene.x_min),
                x_max=float(self.scene.x_max),
                view_radius=float(self.scene.view_radius),
                field_nx=int(self._field_nx),
                field_ny=int(self._field_ny),
                field_nz=int(self._field_nz),
                field_gain=float(field_gain),
                surface_gain=0.0,
                hdr_white_percentile=99.8,
            )

            s_top = np.asarray(s_top_rgb, dtype=np.float32)
            s_side = np.asarray(s_side_rgb, dtype=np.float32)
            f_top = np.asarray(f_top_rgb, dtype=np.float32)
            f_side = np.asarray(f_side_rgb, dtype=np.float32)

            # C++ field projection indexes row 0 at negative world Y/Z, while
            # surface projection and overlay mapping index row 0 at positive
            # world Y/Z. Flip field views so wireframe and volumetric light
            # occupy the same world-space side in both panes.
            f_top = np.ascontiguousarray(f_top[::-1, :, :], dtype=np.float32)
            f_side = np.ascontiguousarray(f_side[::-1, :, :], dtype=np.float32)

            s_leak = float(np.clip(surface_leak, 0.0, 0.999))
            f_leak = float(np.clip(field_leak, 0.0, 0.999))
            s_mix = float(1.0 - s_leak)
            f_mix = float(1.0 - f_leak)

            if self._surface_top_ema is None or self._surface_top_ema.shape != s_top.shape:
                self._surface_top_ema = np.ascontiguousarray(s_top, dtype=np.float32)
                self._surface_side_ema = np.ascontiguousarray(s_side, dtype=np.float32)
            else:
                self._surface_top_ema = np.ascontiguousarray(
                    s_leak * self._surface_top_ema + s_mix * s_top,
                    dtype=np.float32,
                )
                self._surface_side_ema = np.ascontiguousarray(
                    s_leak * self._surface_side_ema + s_mix * s_side,
                    dtype=np.float32,
                )

            if self._field_top_ema is None or self._field_top_ema.shape != f_top.shape:
                self._field_top_ema = np.ascontiguousarray(f_top, dtype=np.float32)
                self._field_side_ema = np.ascontiguousarray(f_side, dtype=np.float32)
            else:
                self._field_top_ema = np.ascontiguousarray(
                    f_leak * self._field_top_ema + f_mix * f_top,
                    dtype=np.float32,
                )
                self._field_side_ema = np.ascontiguousarray(
                    f_leak * self._field_side_ema + f_mix * f_side,
                    dtype=np.float32,
                )

            top = np.clip(self._surface_top_ema + self._field_top_ema, 0.0, 1.0).astype(np.float32, copy=False)
            side = np.clip(self._surface_side_ema + self._field_side_ema, 0.0, 1.0).astype(np.float32, copy=False)
            return top, side

    def display_pipeline_records(
        self,
        field_gain: float = 1.0,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Return field texels and per-class surface-hit vertices from pipeline records.

        Returns:
            field_rgb : float32 (Z, Y, X, 3) — 3D texture for volume raycasting
            verts_by_class : list of 4 float32 (N, 4) arrays, one per display class:
                [0] forward strikes (emissive → surface)
                [1] forward volume  (emissive → wave arena)
                [2] reverse volume  (sensor   → wave arena)
                [3] reverse strikes (sensor   → surface)
            Each row: (x_norm, y_norm, z_norm, amplitude) normalised to [0,1] scene box.
        """
        _empty4 = np.zeros((0, 4), dtype=np.float32)

        with self._trace_lock:
            grid = np.asarray(self.tracer.get_field_capture_grid_reim(), dtype=np.float32)
            if grid.ndim != 3 or grid.shape[2] != 2:
                raise RuntimeError("field capture grid has unexpected shape")
            n_vox = int(self._field_nx * self._field_ny * self._field_nz)
            if int(grid.shape[1]) != n_vox:
                raise RuntimeError("field capture grid size does not match configured volume")

            amp = np.sqrt(np.maximum(0.0, grid[..., 0] ** 2 + grid[..., 1] ** 2))
            # Keep the field texture in raw persistent exposure units.  Do not
            # percentile-normalize per frame: that makes accumulated light appear
            # to fade whenever a newer/brighter voxel changes the white point.
            bands = min(int(amp.shape[0]), int(self.freq_hz.shape[0]))
            wl_nm = (C_LIGHT / np.maximum(np.asarray(self.freq_hz[:bands], dtype=np.float64), EPS)) * 1.0e9
            rgb_w = _wavelength_to_rgb_weights(wl_nm).astype(np.float32)
            rgb_flat = np.einsum("bn,bc->nc", amp[:bands], rgb_w, optimize=True)
            field_rgb = rgb_flat.reshape((self._field_nz, self._field_ny, self._field_nx, 3))
            field_rgb = np.maximum(field_rgb, 0.0) * float(max(0.0, field_gain))

        # ── Build per-class vertex arrays from the pipeline ring buffer ──────
        x_span  = max(EPS, float(self.scene.x_max - self.scene.x_min))
        yz_span = max(EPS, float(2.0 * self.scene.view_radius))
        x_min   = float(self.scene.x_min)
        r       = float(self.scene.view_radius)

        # ── Read raw hit positions from ring buffer; split by display class ──────
        with self._segs_lock:
            count = self._VIS_CAP if self._vis_full else self._vis_ptr
            buf   = self._vis_buf[:count].copy() if count > 0 else None

        verts_by_class: List[np.ndarray] = [_empty4, _empty4, _empty4, _empty4]
        n_by_class = [0, 0, 0, 0]

        if buf is not None and buf.shape[0] > 0:
            # Normalise world positions → [0,1] scene box for the renderer.
            norm_x = np.clip((buf[:, 0] - x_min) / x_span, 0.0, 1.0)
            norm_y = np.clip((buf[:, 1] + r)      / yz_span, 0.0, 1.0)
            norm_z = np.clip((buf[:, 2] + r)      / yz_span, 0.0, 1.0)
            amp_v  = buf[:, 3]
            cls_v  = buf[:, 4].astype(np.int32)

            # Amplitude threshold for volume/field points only (cls 1, 2).
            # Strike points (cls 0, 3) are always kept.
            # Use a fast sample-based estimate instead of a full sort — sorting
            # 20M elements per frame blocks the main thread long enough to
            # trigger Windows TDR and crash the GL context.
            strike_mask = (cls_v == 0) | (cls_v == 3)
            non_strike_amp = amp_v[~strike_mask]
            if non_strike_amp.size > 4096:
                sample = non_strike_amp[::max(1, non_strike_amp.size // 4096)]
                amp_thresh = float(np.percentile(sample, 5))
            elif non_strike_amp.size > 0:
                amp_thresh = float(non_strike_amp.min())
            else:
                amp_thresh = 1.0e-9
            amp_thresh = max(amp_thresh, 1.0e-9)
            keep = strike_mask | (amp_v > amp_thresh)

            for cls in range(4):
                m = keep & (cls_v == cls)
                if not np.any(m):
                    continue
                v = np.empty((int(np.count_nonzero(m)), 4), dtype=np.float32)
                v[:, 0] = norm_x[m]
                v[:, 1] = norm_y[m]
                v[:, 2] = norm_z[m]
                v[:, 3] = amp_v[m]
                verts_by_class[cls] = np.ascontiguousarray(v)
                n_by_class[cls]     = v.shape[0]

        try:
            ps = self.tracer.pipeline_stats()
            stage_parts = []
            for stage in ("t1", "t2", "t3", "t4"):
                st       = ps[stage]
                cpu_tp   = float(st["throughput"])
                gpu_tp   = float(st["gpu_throughput"])
                cpu_n    = int(st["processed"])
                gpu_n    = int(st["gpu_processed"])
                bs_gpu   = int(st["gpu_batch_size"])
                gpu_frac = float(st["gpu_fraction"])
                # Skip stages where neither CPU nor GPU has processed anything yet
                if cpu_n == 0 and gpu_n == 0:
                    continue
                cpu_str = f"cpu={cpu_tp/1e3:.1f}k(n={cpu_n})" if cpu_n > 0 else "cpu=-"
                gpu_str = f"gpu={gpu_tp/1e3:.1f}k(n={gpu_n},bs={bs_gpu},f={gpu_frac:.2f})" if gpu_n > 0 else "gpu=-"
                stage_parts.append(f"{stage}:[{cpu_str} {gpu_str}]")
            pipeline_summary = "  ".join(stage_parts) if stage_parts else "no activity"
        except Exception:
            pipeline_summary = "n/a"

        field_nonzero = int(np.count_nonzero(np.maximum.reduce(field_rgb, axis=3)))
        print(
            "[display-records]",
            f"field_nonzero={field_nonzero}",
            f"pts={[n_by_class[c] for c in range(4)]}",
            f"pipeline={pipeline_summary}",
            flush=True,
        )

        return (
            np.ascontiguousarray(np.maximum(field_rgb, 0.0).astype(np.float32)),
            verts_by_class,
        )


def map_top(scene: SceneConfig, x: float, y: float, rect: pygame.Rect) -> Tuple[int, int]:
    u = (x - scene.x_min) / max(EPS, (scene.x_max - scene.x_min))
    v = 0.5 - (y / max(EPS, scene.view_radius * 2.0))
    px = rect.left + int(np.rint(np.clip(u, 0.0, 1.0) * (rect.width - 1)))
    py = rect.top + int(np.rint(np.clip(v, 0.0, 1.0) * (rect.height - 1)))
    return px, py


def map_side(scene: SceneConfig, x: float, z: float, rect: pygame.Rect) -> Tuple[int, int]:
    return map_top(scene, x, z, rect)


def draw_scene_overlays(
    surf: pygame.Surface,
    rect: pygame.Rect,
    scene: SceneConfig,
    top_view: bool,
    clear_bg: bool = True,
    sidecar: FreeFrequencySidecar | None = None,
    tri_vertices: np.ndarray | None = None,
    wireframe: bool = False,
) -> None:
    if clear_bg:
        pygame.draw.rect(surf, (15, 16, 18), rect)
    pygame.draw.rect(surf, (64, 66, 72), rect, 1)

    def line_x(xv: float, col: Tuple[int, int, int]) -> None:
        p0 = map_top(scene, xv, -scene.view_radius, rect)
        p1 = map_top(scene, xv, +scene.view_radius, rect)
        pygame.draw.line(surf, col, p0, p1, 1)

    if wireframe and tri_vertices is not None:
        tri = np.asarray(tri_vertices, dtype=np.float64)
        if tri.ndim == 3 and tri.shape[1:] == (3, 3):
            col = (86, 94, 106)
            step = max(1, int(tri.shape[0] // 2600))
            for t in tri[::step]:
                if top_view:
                    p0 = map_top(scene, float(t[0, 0]), float(t[0, 1]), rect)
                    p1 = map_top(scene, float(t[1, 0]), float(t[1, 1]), rect)
                    p2 = map_top(scene, float(t[2, 0]), float(t[2, 1]), rect)
                else:
                    p0 = map_side(scene, float(t[0, 0]), float(t[0, 2]), rect)
                    p1 = map_side(scene, float(t[1, 0]), float(t[1, 2]), rect)
                    p2 = map_side(scene, float(t[2, 0]), float(t[2, 2]), rect)
                pygame.draw.line(surf, col, p0, p1, 1)
                pygame.draw.line(surf, col, p1, p2, 1)
                pygame.draw.line(surf, col, p2, p0, 1)
    else:
        if top_view:
            p0 = map_top(scene, scene.tube_x0, scene.tube_radius, rect)
            p1 = map_top(scene, scene.tube_x1, scene.tube_radius, rect)
            p2 = map_top(scene, scene.tube_x0, -scene.tube_radius, rect)
            p3 = map_top(scene, scene.tube_x1, -scene.tube_radius, rect)
        else:
            p0 = map_side(scene, scene.tube_x0, scene.tube_radius, rect)
            p1 = map_side(scene, scene.tube_x1, scene.tube_radius, rect)
            p2 = map_side(scene, scene.tube_x0, -scene.tube_radius, rect)
            p3 = map_side(scene, scene.tube_x1, -scene.tube_radius, rect)

        pygame.draw.line(surf, (68, 96, 120), p0, p1, 1)
        pygame.draw.line(surf, (68, 96, 120), p2, p3, 1)

        for lens in _scene_lenses(scene):
            lens_r = float(lens.aperture_radius)
            if top_view:
                lf0 = map_top(scene, lens.x_front, -lens_r, rect)
                lf1 = map_top(scene, lens.x_front, +lens_r, rect)
                lb0 = map_top(scene, lens.x_back, -lens_r, rect)
                lb1 = map_top(scene, lens.x_back, +lens_r, rect)
            else:
                lf0 = map_side(scene, lens.x_front, -lens_r, rect)
                lf1 = map_side(scene, lens.x_front, +lens_r, rect)
                lb0 = map_side(scene, lens.x_back, -lens_r, rect)
                lb1 = map_side(scene, lens.x_back, +lens_r, rect)
            pygame.draw.line(surf, (168, 206, 232), lf0, lf1, 2)
            pygame.draw.line(surf, (168, 206, 232), lb0, lb1, 2)

    line_x(scene.image_plate.x, (72, 140, 88))

    if sidecar is not None and sidecar.freq_hz.size > 1:
        ys = np.linspace(-0.9 * scene.object_plane.radius, 0.9 * scene.object_plane.radius, int(sidecar.freq_hz.size), dtype=np.float64)
        for yv, fv in zip(ys, sidecar.freq_hz):
            col = _freq_to_hue_rgb(float(fv))
            p = map_top(scene, scene.object_plane.x, float(yv), rect) if top_view else map_side(scene, scene.object_plane.x, float(yv), rect)
            pygame.draw.circle(surf, col, p, 2)
        # Stretch-goal visual aid: draw exact lens curvature profiles in side view.
        if not top_view:
            for lens in _scene_lenses(scene):
                rr = np.linspace(-float(lens.aperture_radius), float(lens.aperture_radius), 48, dtype=np.float64)
                fr = _lens_front_x(lens, np.abs(rr))
                br = _lens_back_x(lens, np.abs(rr))
                front_pts = [map_side(scene, float(xv), float(zv), rect) for xv, zv in zip(fr, rr)]
                back_pts = [map_side(scene, float(xv), float(zv), rect) for xv, zv in zip(br, rr)]
                if len(front_pts) >= 2:
                    pygame.draw.lines(surf, (154, 200, 230), False, front_pts, 1)
                if len(back_pts) >= 2:
                    pygame.draw.lines(surf, (154, 200, 230), False, back_pts, 1)

    else:
        src_t = map_top(scene, scene.object_plane.x, 0.0, rect) if top_view else map_side(scene, scene.object_plane.x, 0.0, rect)
        pygame.draw.circle(surf, (255, 188, 98), src_t, 4)


# ── Cross-section triangle kind constants ──────────────────────────────── #

TRI_KIND_DEFAULT  = 0   # walls, structural
TRI_KIND_LENS     = 1   # transmissive glass
TRI_KIND_APERTURE = 2   # aperture / blade
TRI_KIND_EMISSIVE = 3   # light source surface
TRI_KIND_SENSOR   = 4   # image sensor / detector plate

# RGBA fill colours per kind — lens is semi-transparent so field shows through
_TRI_FILL_RGBA: Dict[int, Tuple[int, int, int, int]] = {
    TRI_KIND_DEFAULT:  ( 50,  58,  74, 210),
    TRI_KIND_LENS:     (150, 210, 255, 110),
    TRI_KIND_APERTURE: (  8,   8,  10, 255),
    TRI_KIND_EMISSIVE: (255, 210,  70, 210),
    TRI_KIND_SENSOR:   (160, 230, 160, 190),
}


def _compute_xs_polygons(
    W: int, H: int,
    scene: SceneConfig,
    tri_vertices: np.ndarray,   # (N, 3, 3) float64 — caller passes a .copy()
    tri_kind: np.ndarray,       # (N,) int8
    top_view: bool,
    clip: float,
):
    """Thread-safe: pure numpy geometry — no pygame calls.

    Returns (pu, pv, colors) sorted back-to-front, or (None, None, None).
      pu, pv : (M, 3) int32  pixel coords of each triangle vertex
      colors : (M, 4) uint8  RGBA per triangle
    Main thread draws the polygons incrementally via pygame.draw.polygon.
    """
    n = int(tri_vertices.shape[0])
    if n == 0:
        return None, None, None

    depth_ax = 1 if top_view else 2
    depths   = tri_vertices[:, :, depth_ax].mean(axis=1)   # (N,) vectorised
    mask     = depths <= clip
    if not mask.any():
        return None, None, None

    vis_tri  = tri_vertices[mask]
    vis_kind = tri_kind[mask]
    vis_dep  = depths[mask]

    MAX_POLYS = 4000
    if len(vis_tri) > MAX_POLYS:
        nd      = vis_kind != TRI_KIND_DEFAULT
        nd_idx  = np.where(nd)[0]
        def_idx = np.where(~nd)[0]
        budget  = max(0, MAX_POLYS - int(nd_idx.size))
        if def_idx.size > budget:
            step    = max(1, def_idx.size // max(1, budget))
            def_idx = def_idx[::step]
        keep     = np.concatenate([nd_idx, def_idx])
        vis_tri  = vis_tri[keep]
        vis_kind = vis_kind[keep]
        vis_dep  = vis_dep[keep]

    order    = np.argsort(vis_dep)          # ascending = farthest first
    vis_tri  = vis_tri[order]
    vis_kind = vis_kind[order]

    x_span = max(EPS, scene.x_max - scene.x_min)
    y_span = max(EPS, scene.view_radius * 2.0)
    va     = 1 if top_view else 2           # vertex component for vertical axis

    # Batch-project all vertices → pixel space (fully vectorised)
    pu = ((vis_tri[:, :, 0] - scene.x_min) / x_span) * (W - 1)
    pv = (0.5 - vis_tri[:, :, va] / y_span) * (H - 1)
    pu = np.clip(pu, 0, W - 1).astype(np.int32)    # (M, 3)
    pv = np.clip(pv, 0, H - 1).astype(np.int32)    # (M, 3)

    # Pre-build RGBA color array (one lookup per triangle, not per pixel)
    default_col = _TRI_FILL_RGBA[TRI_KIND_DEFAULT]
    colors = np.array(
        [_TRI_FILL_RGBA.get(int(k), default_col) for k in vis_kind],
        dtype=np.uint8,
    )   # (M, 4)

    return pu, pv, colors


def _build_bdpt_path_segments(
    records: Optional[np.ndarray],
    sensor_group_id: int,
    sensor_origin_xyz: Optional[np.ndarray] = None,
    max_subpaths: int = 700,
    max_segments: int = 14_000,
    fallback_stub_length_m: float = 0.35,
) -> np.ndarray:
    """Build line segments from backend EndpointRecord rows for ray overlay."""
    if records is None:
        return np.zeros((0, 2, 3), dtype=np.float32)
    rec = np.asarray(records, dtype=np.float32)
    if rec.ndim != 2 or rec.shape[1] < 8 or rec.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)

    try:
        col_i32 = rec.view(np.int32).reshape(rec.shape[0], rec.shape[1])
    except Exception:
        return np.zeros((0, 2, 3), dtype=np.float32)

    subpath_id = col_i32[:, 0].astype(np.int64, copy=False)
    band_id = col_i32[:, 1].astype(np.int64, copy=False)
    group_id = col_i32[:, 2].astype(np.int64, copy=False)
    vertex_index = col_i32[:, 3].astype(np.int64, copy=False)
    pathlen = np.asarray(rec[:, 7], dtype=np.float64)
    pos = np.asarray(rec[:, 4:7], dtype=np.float64)
    dirs = np.asarray(rec[:, 8:11], dtype=np.float64) if rec.shape[1] >= 11 else None

    use_sensor_anchor = sensor_origin_xyz is not None
    sensor_origin = None
    if use_sensor_anchor:
        try:
            sensor_origin = np.asarray(sensor_origin_xyz, dtype=np.float64).reshape(3)
        except Exception:
            sensor_origin = None

    # Use only true sensor-capture records for reverse-overlay reconstruction.
    if sensor_group_id < 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    keep = (group_id == int(sensor_group_id))
    if not np.any(keep):
        return np.zeros((0, 2, 3), dtype=np.float32)

    # Only use launched PIXEL_CONE samples; ignore negative sentinels.
    keep &= (vertex_index >= 0)
    if not np.any(keep):
        return np.zeros((0, 2, 3), dtype=np.float32)

    # Avoid stitching across spectral duplicates by selecting a single band.
    band_min = int(np.min(band_id[keep]))
    keep &= (band_id == band_min)

    idx = np.flatnonzero(keep)
    if idx.size < 1:
        return np.zeros((0, 2, 3), dtype=np.float32)

    order = np.lexsort((pathlen[idx], vertex_index[idx], subpath_id[idx]))
    idx = idx[order]

    segs: List[np.ndarray] = []
    i = 0
    while i < idx.size and len(segs) < int(max_segments):
        sid = int(subpath_id[idx[i]])
        vid = int(vertex_index[idx[i]])
        j = i + 1
        while j < idx.size and int(subpath_id[idx[j]]) == sid and int(vertex_index[idx[j]]) == vid:
            j += 1
        if (j - i) >= 1:
            pp = pos[idx[i:j], :]
            # Collapse repeated points to keep paths monotonic and avoid
            # inter-band/inter-hit duplicates creating false zig-zags.
            if pp.shape[0] > 1:
                unique_pts: List[np.ndarray] = [pp[0]]
                for p in pp[1:]:
                    if float(np.linalg.norm(p - unique_pts[-1])) >= 1.0e-7:
                        unique_pts.append(p)
                pp = np.asarray(unique_pts, dtype=np.float64)
            if sensor_origin is not None and pp.shape[0] > 0:
                p_first = pp[0]
                if (not np.any(np.isnan(p_first))) and float(np.linalg.norm(p_first - sensor_origin)) >= 1.0e-7:
                    segs.append(np.stack([sensor_origin, p_first], axis=0))
                    if len(segs) >= int(max_segments):
                        break
            for k in range(pp.shape[0] - 1):
                p0 = pp[k]
                p1 = pp[k + 1]
                if np.any(np.isnan(p0)) or np.any(np.isnan(p1)):
                    continue
                if float(np.linalg.norm(p1 - p0)) < 1.0e-7:
                    continue
                segs.append(np.stack([p0, p1], axis=0))
                if len(segs) >= int(max_segments):
                    break
        i = j

    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(segs, dtype=np.float32), dtype=np.float32)


def draw_bdpt_segments_overlay(
    surf: pygame.Surface,
    rect: pygame.Rect,
    scene: SceneConfig,
    segments_xyz: np.ndarray,
    top_view: bool,
) -> None:
    if segments_xyz.ndim != 3 or segments_xyz.shape[1:] != (2, 3) or segments_xyz.shape[0] <= 0:
        return
    n_seg = int(segments_xyz.shape[0])
    for i, s in enumerate(segments_xyz):
        t = float(i) / float(max(1, n_seg - 1))
        # Gradient encodes progression along reconstructed reverse paths.
        col = (
            int(255.0 * (1.0 - 0.55 * t)),
            int(180.0 + 70.0 * t),
            int(96.0 + 120.0 * t),
        )
        if top_view:
            p0 = map_top(scene, float(s[0, 0]), float(s[0, 1]), rect)
            p1 = map_top(scene, float(s[1, 0]), float(s[1, 1]), rect)
        else:
            p0 = map_side(scene, float(s[0, 0]), float(s[0, 2]), rect)
            p1 = map_side(scene, float(s[1, 0]), float(s[1, 2]), rect)
        pygame.draw.line(surf, col, p0, p1, 2)


def draw_bdpt_transition_markers(
    surf: pygame.Surface,
    rect: pygame.Rect,
    scene: SceneConfig,
    segments_xyz: np.ndarray,
    top_view: bool,
    min_turn_deg: float = 18.0,
) -> int:
    """Mark sharp direction changes as likely material transition events."""
    if segments_xyz.ndim != 3 or segments_xyz.shape[1:] != (2, 3) or int(segments_xyz.shape[0]) < 2:
        return 0

    threshold = math.cos(math.radians(float(min_turn_deg)))
    marked = 0
    for i in range(int(segments_xyz.shape[0]) - 1):
        a0 = segments_xyz[i, 0]
        a1 = segments_xyz[i, 1]
        b0 = segments_xyz[i + 1, 0]
        b1 = segments_xyz[i + 1, 1]

        # Only compare adjacent segments that share an endpoint.
        if float(np.linalg.norm(a1 - b0)) > 2.0e-4:
            continue

        va = a1 - a0
        vb = b1 - b0
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na < 1.0e-9 or nb < 1.0e-9:
            continue

        cos_turn = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
        if cos_turn >= threshold:
            continue

        if top_view:
            p = map_top(scene, float(a1[0]), float(a1[1]), rect)
        else:
            p = map_side(scene, float(a1[0]), float(a1[2]), rect)
        pygame.draw.circle(surf, (88, 240, 255), p, 2)
        marked += 1
    return marked


def visualize_bdpt_rays_3d_export(
    bdpt_records: Optional[np.ndarray],
    scene: SceneConfig,
    output_file: str = "bdpt_rays_overlay.txt",
    max_rays: int = 1000,
) -> Dict[str, any]:
    """Export BDPT ray paths to a text file for 3D visualization.
    
    Creates a detailed breakdown of where each backward ray travels,
    including bounce points, materials hit, and amplitude decay.
    
    Args:
        bdpt_records: EndpointRecord array from BDPT (None returns empty result)
        scene: SceneConfig for reference positions
        output_file: Output text file path
        max_rays: Maximum number of rays to export (for clarity)
    
    Returns:
        Dict with statistics: ray_count, total_length, success_count, bounce_distribution
    """
    result = {
        "ray_count": 0,
        "total_path_length": 0.0,
        "endpoint_count": 0,
        "max_path_length": 0.0,
        "min_path_length": 0.0,
        "bounce_distribution": {},
    }
    
    if bdpt_records is None:
        return result
    
    rec = np.asarray(bdpt_records, dtype=np.float32)
    if rec.ndim != 2 or rec.shape[0] < 1:
        return result
    
    # EndpointRecord layout: subpath_id, band_id, group_id, vertex_idx, pos[3], pathlen, dir[3], pdf, amp_re, amp_im, cos_theta, _pad
    # We need: pos[4:7], pathlen[7], amp[9:11]
    try:
        with open(output_file, "w") as f:
            f.write("# BDPT Ray Path Visualization Data\n")
            f.write("# Format: subpath_id pos_x pos_y pos_z pathlen amplitude\n")
            
            subpath_ids = rec[:, 0].astype(np.int32)
            positions = rec[:, 4:7].astype(np.float32)
            pathlens = rec[:, 7].astype(np.float32)
            amplitudes = np.sqrt(rec[:, 9]**2 + rec[:, 10]**2).astype(np.float32)  # |amp|
            
            unique_subpaths = np.unique(subpath_ids)
            if unique_subpaths.size > max_rays:
                stride = max(1, unique_subpaths.size // max_rays)
                unique_subpaths = unique_subpaths[::stride]
            
            ray_count = 0
            for subpath_id in unique_subpaths:
                mask = (subpath_ids == subpath_id)
                if not np.any(mask):
                    continue
                
                ray_count += 1
                indices = np.where(mask)[0]
                positions_ray = positions[indices]
                pathlens_ray = pathlens[indices]
                amplitudes_ray = amplitudes[indices]
                
                f.write(f"\nray {ray_count} subpath={int(subpath_id)} bounces={len(indices)}\n")
                
                # Write each bounce point
                for i, (pos, plen, amp) in enumerate(zip(positions_ray, pathlens_ray, amplitudes_ray)):
                    f.write(f"  bounce {i}: pos=({float(pos[0]):.6f}, {float(pos[1]):.6f}, {float(pos[2]):.6f}) ")
                    f.write(f"pathlen={float(plen):.6f} amplitude={float(amp):.6e}\n")
                
                result["endpoint_count"] += len(indices)
                result["total_path_length"] += float(pathlens_ray[-1]) if len(pathlens_ray) > 0 else 0.0
                result["max_path_length"] = max(result["max_path_length"], float(pathlens_ray.max()) if len(pathlens_ray) > 0 else 0.0)
                if ray_count == 1:
                    result["min_path_length"] = float(pathlens_ray[-1]) if len(pathlens_ray) > 0 else 0.0
                else:
                    if len(pathlens_ray) > 0:
                        result["min_path_length"] = min(result["min_path_length"], float(pathlens_ray[-1]))
                
                # Track bounce distribution
                bounce_count = len(indices)
                result["bounce_distribution"][str(bounce_count)] = result["bounce_distribution"].get(str(bounce_count), 0) + 1
            
            f.write("\n# Statistics\n")
            f.write(f"# Total rays: {ray_count}\n")
            f.write(f"# Total endpoints: {result['endpoint_count']}\n")
            f.write(f"# Avg path length: {result['total_path_length'] / max(1, ray_count):.6f}\n")
            f.write(f"# Max path length: {result['max_path_length']:.6f}\n")
            
            result["ray_count"] = int(ray_count)
            
    except Exception as e:
        print(f"[ray-export] Error writing ray file: {e}")
    
    return result


def rgb_to_surface(rgb_hw3: np.ndarray, size: Tuple[int, int]) -> pygame.Surface:
    arr = np.ascontiguousarray(np.clip(np.asarray(rgb_hw3, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    surf = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGB")
    return pygame.transform.smoothscale(surf, size)


def _clone_scene_with_lenses(scene: SceneConfig, lenses: Sequence[LensConfig]) -> SceneConfig:
    return SceneConfig(
        x_min=scene.x_min,
        x_max=scene.x_max,
        source_x=scene.source_x,
        source_radius=scene.source_radius,
        tube_x0=scene.tube_x0,
        tube_x1=scene.tube_x1,
        tube_radius=scene.tube_radius,
        baffle0_x=scene.baffle0_x,
        baffle0_aperture=scene.baffle0_aperture,
        baffle1_x=scene.baffle1_x,
        baffle1_aperture=scene.baffle1_aperture,
        baffle2_x=scene.baffle2_x,
        baffle2_aperture=scene.baffle2_aperture,
        screen_x=scene.screen_x,
        screen_radius=scene.screen_radius,
        view_radius=scene.view_radius,
        enable_debug_plate=scene.enable_debug_plate,
        debug_plate_x=scene.debug_plate_x,
        debug_plate_radius=scene.debug_plate_radius,
        object_plane=scene.object_plane,
        image_plate=scene.image_plate,
        side_room_y=scene.side_room_y,
        side_room_aperture=scene.side_room_aperture,
        side_room_bore_radius=scene.side_room_bore_radius,
        side_room_source_radius=scene.side_room_source_radius,
        side_room_wall_outer=scene.side_room_wall_outer,
        side_room_depth=scene.side_room_depth,
        side_room_source_offset=scene.side_room_source_offset,
        side_room_source_emission=scene.side_room_source_emission,
        side_room_diffuser_enabled=scene.side_room_diffuser_enabled,
        side_room_diffuser_radius=scene.side_room_diffuser_radius,
        side_room_diffuser_thickness=scene.side_room_diffuser_thickness,
        side_room_diffuser_transmittance=scene.side_room_diffuser_transmittance,
        side_room_diffuser_diffuse_frac=scene.side_room_diffuser_diffuse_frac,
        side_room_diffuser_tilt_x_deg=scene.side_room_diffuser_tilt_x_deg,
        side_room_diffuser_tilt_z_deg=scene.side_room_diffuser_tilt_z_deg,
        side_room_source_x=scene.side_room_source_x,
        side_room_source_z=scene.side_room_source_z,
        stage_probe_emitter_enabled=scene.stage_probe_emitter_enabled,
        stage_probe_x=scene.stage_probe_x,
        stage_probe_y=scene.stage_probe_y,
        stage_probe_z=scene.stage_probe_z,
        stage_probe_radius=scene.stage_probe_radius,
        disable_optics=scene.disable_optics,
        lens=lenses[0] if lenses else scene.lens,
        lens_stack=list(lenses),
    )


def _mutate_lenses(lenses: Sequence[LensConfig], rng: np.random.Generator, scale: float) -> List[LensConfig]:
    out: List[LensConfig] = []
    prev_x = 0.86
    for lens in lenses:
        cx = float(np.clip(lens.center_x + rng.normal(0.0, 0.030 * scale), prev_x + 0.055, 1.58))
        thickness = float(np.clip(lens.thickness + rng.normal(0.0, 0.006 * scale), 0.024, 0.060))
        aperture = float(np.clip(lens.aperture_radius + rng.normal(0.0, 0.006 * scale), 0.020, 0.052))
        rf = float(np.clip(lens.radius_front + rng.normal(0.0, 0.030 * scale), aperture + 0.035, 0.240))
        rb = float(np.clip(lens.radius_back + rng.normal(0.0, 0.030 * scale), aperture + 0.035, 0.240))
        cand = LensConfig(cx, thickness, aperture, rf, rb, lens.ior)
        if not _lens_is_valid(cand):
            cand = LensConfig(cx, thickness, min(aperture, 0.040), max(rf, 0.110), max(rb, 0.110), lens.ior)
        out.append(cand)
        prev_x = cand.x_back
    return out


def _score_scene_spot(scene: SceneConfig, rays_per_emitter: int = 24, seed: int = 1234) -> Tuple[float, dict]:
    sidecar = FreeFrequencySidecar.lazy_prepare(4)
    bench = ForwardCppLensBench(scene=scene, freq_hz=DEFAULT_FREQ_HZ[:4].copy(), view_h=96, view_w=192, sidecar=sidecar)
    records = bench.trace_forward(
        rays_per_emitter=max(1, int(rays_per_emitter)),
        seed=int(seed),
        max_bounces=64,
        decay=0.0,
    )
    plate = bench.capture_plate_rgb(64)
    ideal = bench.ideal_object_rgb(64)
    plate_l = np.sum(plate.astype(np.float64), axis=2)
    ideal_l = np.sum(ideal.astype(np.float64), axis=2)
    throughput = float(np.sum(plate_l))
    if throughput <= EPS:
        return 1.0e6, {"records": int(records), "mse": float("inf"), "throughput": 0.0}
    plate_l /= max(EPS, float(np.max(plate_l)))
    ideal_l /= max(EPS, float(np.max(ideal_l)))
    mse = float(np.mean((plate_l - ideal_l) ** 2))
    score = mse + 0.010 / math.sqrt(max(1.0, throughput))
    return score, {"records": int(records), "mse": mse, "throughput": throughput}


def tune_zoom_focus_scene(base_scene: SceneConfig, iterations: int = 32, seed: int = 20260512) -> SceneConfig:
    rng = np.random.default_rng(int(seed))
    best_lenses = list(_scene_lenses(base_scene))
    best_scene = _clone_scene_with_lenses(base_scene, best_lenses)
    best_score, best_meta = _score_scene_spot(best_scene, seed=int(rng.integers(1, 2**31 - 1)))
    print(f"[tune] start score={best_score:.6g} mse={best_meta['mse']:.5f} records={best_meta['records']}", flush=True)
    for i in range(max(1, int(iterations))):
        scale = max(0.20, 1.0 - i / max(1, int(iterations)))
        cand_lenses = _mutate_lenses(best_lenses, rng, scale)
        cand_scene = _clone_scene_with_lenses(base_scene, cand_lenses)
        score, meta = _score_scene_spot(cand_scene, seed=int(rng.integers(1, 2**31 - 1)))
        if score < best_score:
            best_score, best_meta, best_lenses, best_scene = score, meta, cand_lenses, cand_scene
            print(f"[tune] best i={i:03d} score={best_score:.6g} mse={meta['mse']:.5f} records={meta['records']}", flush=True)
    return best_scene


class AsyncTraceAccumulator:
    """Background tracer runner with bounded queue back pressure."""

    def __init__(
        self,
        bench: ForwardCppLensBench,
        *,
        max_bounces: int,
        decay: float = 0.97,
        max_queue: int = 3,
    ) -> None:
        self.bench = bench
        self.max_bounces = int(max_bounces)
        self.decay = float(decay)
        self._q: queue.Queue[Tuple[int, int]] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self.last_records = 0
        self.completed_batches = 0

    def start(self) -> None:
        if self._thr is not None and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, name="lens-trace-worker", daemon=True)
        self._thr.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=max(0.0, float(timeout_s)))

    def pending(self) -> int:
        return int(self._q.qsize())

    def submit(self, rays_per_emitter: int, seed: int, timeout_s: float = 0.01) -> bool:
        try:
            self._q.put((int(rays_per_emitter), int(seed)), timeout=max(0.0, float(timeout_s)))
            return True
        except queue.Full:
            return False

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                rays_per_emitter, seed = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.last_records = int(
                    self.bench.trace_forward(
                        rays_per_emitter=rays_per_emitter,
                        seed=seed,
                        max_bounces=self.max_bounces,
                        decay=self.decay,
                    )
                )
                self.completed_batches += 1
            finally:
                self._q.task_done()


def _project_lens_centers(
    raw_centers: np.ndarray,
    lenses: Sequence[LensConfig],
    scene: SceneConfig,
    min_gap: float = 0.004,
) -> np.ndarray:
    c = np.ascontiguousarray(np.asarray(raw_centers, dtype=np.float64).reshape(-1), dtype=np.float64).copy()
    n = int(c.size)
    if n <= 0:
        return c
    thickness = np.asarray([float(l.thickness) for l in lenses], dtype=np.float64)
    lower = np.asarray(scene.tube_x0 + 0.005 + 0.5 * thickness, dtype=np.float64)
    upper = np.asarray(scene.tube_x1 - 0.005 - 0.5 * thickness, dtype=np.float64)
    c = np.clip(c, lower, upper)
    for i in range(1, n):
        min_ci = c[i - 1] + 0.5 * thickness[i - 1] + 0.5 * thickness[i] + float(min_gap)
        if c[i] < min_ci:
            c[i] = min_ci
    for i in range(n - 2, -1, -1):
        max_ci = c[i + 1] - (0.5 * thickness[i] + 0.5 * thickness[i + 1] + float(min_gap))
        if c[i] > max_ci:
            c[i] = max_ci
    return np.ascontiguousarray(np.clip(c, lower, upper), dtype=np.float64)


def _score_lens_centers(
    base_scene: SceneConfig,
    centers: np.ndarray,
    *,
    rays_per_emitter: int,
    seed: int,
    pixels: int = 56,
    use_bdpt: bool = False,
) -> float:
    base_lenses = _scene_lenses(base_scene)
    cand_lenses = [
        LensConfig(float(cx), l.thickness, l.aperture_radius, l.radius_front, l.radius_back, l.ior)
        for cx, l in zip(np.asarray(centers, dtype=np.float64), base_lenses)
    ]
    scene = _clone_scene_with_lenses(base_scene, cand_lenses)
    sidecar = FreeFrequencySidecar.lazy_prepare(4)
    bench = ForwardCppLensBench(scene=scene, freq_hz=DEFAULT_FREQ_HZ[:4].copy(), view_h=96, view_w=192, sidecar=sidecar)
    bench.trace_forward(
        rays_per_emitter=max(1, int(rays_per_emitter)),
        seed=int(seed),
        max_bounces=24,
        decay=0.0,
    )
    if use_bdpt:
        plate = bench.capture_plate_bdpt_rgb(
            int(pixels),
            aperture_samples=1,
            seed=int(seed + 101),
            max_bounces=24,
        ).astype(np.float64)
    else:
        plate = bench.capture_plate_rgb(int(pixels)).astype(np.float64)
    ideal = bench.ideal_object_rgb(int(pixels)).astype(np.float64)
    plate_l = np.sum(plate, axis=2)
    ideal_l = np.sum(ideal, axis=2)
    pmax = float(np.max(plate_l))
    imax = float(np.max(ideal_l))
    if pmax <= EPS or imax <= EPS:
        return 1.0e6
    plate_l = plate_l / pmax
    ideal_l = ideal_l / imax
    mse = float(np.mean((plate_l - ideal_l) ** 2))
    throughput = float(np.sum(plate_l))
    return mse + 0.008 / math.sqrt(max(1.0, throughput))


def optimize_lens_positions_adam(
    base_scene: SceneConfig,
    *,
    steps: int = 10,
    lr: float = 0.04,
    fd_eps: float = 0.002,
    rays_per_emitter: int = 8,
    seed: int = 20260512,
) -> SceneConfig:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install with: pip install torch")

    base_lenses = _scene_lenses(base_scene)
    c0 = np.asarray([float(l.center_x) for l in base_lenses], dtype=np.float64)
    centers = torch.nn.Parameter(torch.as_tensor(c0, dtype=torch.float64))
    optim = torch.optim.Adam([centers], lr=float(lr))

    best_centers = c0.copy()
    best_loss = float("inf")
    for step in range(max(1, int(steps))):
        with torch.no_grad():
            projected = _project_lens_centers(centers.detach().cpu().numpy(), base_lenses, base_scene)
            centers.copy_(torch.as_tensor(projected, dtype=centers.dtype))

        base_np = centers.detach().cpu().numpy().astype(np.float64, copy=False)
        loss0 = _score_lens_centers(
            base_scene,
            base_np,
            rays_per_emitter=int(rays_per_emitter),
            seed=int(seed + step * 31),
            use_bdpt=True,
        )
        if loss0 < best_loss:
            best_loss = float(loss0)
            best_centers = base_np.copy()

        grads = np.zeros_like(base_np)
        for i in range(int(base_np.size)):
            delta = np.zeros_like(base_np)
            delta[i] = float(fd_eps)
            lp = _score_lens_centers(
                base_scene,
                _project_lens_centers(base_np + delta, base_lenses, base_scene),
                rays_per_emitter=int(rays_per_emitter),
                seed=int(seed + step * 31 + i * 2 + 1),
                use_bdpt=True,
            )
            lm = _score_lens_centers(
                base_scene,
                _project_lens_centers(base_np - delta, base_lenses, base_scene),
                rays_per_emitter=int(rays_per_emitter),
                seed=int(seed + step * 31 + i * 2 + 2),
                use_bdpt=True,
            )
            grads[i] = (float(lp) - float(lm)) / max(float(2.0 * fd_eps), EPS)

        optim.zero_grad(set_to_none=True)
        centers.grad = torch.as_tensor(grads, dtype=centers.dtype)
        optim.step()
        print(f"[adam] step={step:03d} loss={loss0:.6f} best={best_loss:.6f}", flush=True)

    tuned_lenses = [
        LensConfig(float(cx), l.thickness, l.aperture_radius, l.radius_front, l.radius_back, l.ior)
        for cx, l in zip(best_centers, base_lenses)
    ]
    return _clone_scene_with_lenses(base_scene, tuned_lenses)


def _online_adam_update(
    base_scene: SceneConfig,
    centers: "torch.nn.Parameter",
    optim: "torch.optim.Optimizer",
    *,
    step_idx: int,
    fd_eps: float,
    rays_per_emitter: int,
    seed: int,
) -> Tuple[SceneConfig, float]:
    base_lenses = _scene_lenses(base_scene)
    with torch.no_grad():
        projected = _project_lens_centers(centers.detach().cpu().numpy(), base_lenses, base_scene)
        centers.copy_(torch.as_tensor(projected, dtype=centers.dtype))

    base_np = centers.detach().cpu().numpy().astype(np.float64, copy=False)
    loss0 = _score_lens_centers(
        base_scene,
        base_np,
        rays_per_emitter=int(rays_per_emitter),
        seed=int(seed + step_idx * 23),
        use_bdpt=True,
    )
    grads = np.zeros_like(base_np)
    for i in range(int(base_np.size)):
        delta = np.zeros_like(base_np)
        delta[i] = float(fd_eps)
        lp = _score_lens_centers(
            base_scene,
            _project_lens_centers(base_np + delta, base_lenses, base_scene),
            rays_per_emitter=int(rays_per_emitter),
            seed=int(seed + step_idx * 23 + i * 2 + 1),
            use_bdpt=True,
        )
        lm = _score_lens_centers(
            base_scene,
            _project_lens_centers(base_np - delta, base_lenses, base_scene),
            rays_per_emitter=int(rays_per_emitter),
            seed=int(seed + step_idx * 23 + i * 2 + 2),
            use_bdpt=True,
        )
        grads[i] = (float(lp) - float(lm)) / max(float(2.0 * fd_eps), EPS)

    optim.zero_grad(set_to_none=True)
    centers.grad = torch.as_tensor(grads, dtype=centers.dtype)
    optim.step()
    with torch.no_grad():
        projected = _project_lens_centers(centers.detach().cpu().numpy(), base_lenses, base_scene)
        centers.copy_(torch.as_tensor(projected, dtype=centers.dtype))
    tuned_lenses = [
        LensConfig(float(cx), l.thickness, l.aperture_radius, l.radius_front, l.radius_back, l.ior)
        for cx, l in zip(centers.detach().cpu().numpy().astype(np.float64, copy=False), base_lenses)
    ]
    return _clone_scene_with_lenses(base_scene, tuned_lenses), float(loss0)



def _fly_norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _fly_lookat(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _fly_norm(center - eye)
    r = _fly_norm(np.cross(up, f))  # cross(up, f) gives +right when looking along +Z
    u = np.cross(f, r)
    return np.array([
        [ r[0],  r[1],  r[2], -float(np.dot(r, eye))],
        [ u[0],  u[1],  u[2], -float(np.dot(u, eye))],
        [-f[0], -f[1], -f[2],  float(np.dot(f, eye))],
        [    0,      0,     0,                      1],
    ], dtype=np.float32)


def _fly_persp(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    return np.array([
        [f / aspect, 0,  0,                          0                        ],
        [0,          f,  0,                          0                        ],
        [0,          0,  (far + near) / (near - far), 2*far*near / (near - far)],
        [0,          0, -1,                          0                        ],
    ], dtype=np.float32)


def run(
    ray_mode: str = "bdpt",
    sensor_res: int = 64,
    sensor_amp_gain: float = 1.0,
    sensor_min_amplitude: float = 0.0,
    emitter_amp_gain: float = 1.0,
    compute_mode: str = "gpu",
    profile: bool = False,
    field_capture: bool = True,
) -> None:
    try:
        from OpenGL.GL import (
            glGenTextures, glBindTexture, GL_TEXTURE_2D, GL_TEXTURE_3D,
            GL_TEXTURE_2D_ARRAY,
            glTexImage2D, glTexImage3D, GL_RGB, GL_RGBA, GL_RGB32F, GL_RGBA16F, GL_FLOAT,
            glTexParameteri,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
            GL_LINEAR, GL_CLAMP_TO_EDGE, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
            glEnable, glDisable, glClearColor, glClear, GL_COLOR_BUFFER_BIT,
            GL_BLEND, GL_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
            glViewport,
            glEnableClientState, glDisableClientState, glVertexPointer, glDrawArrays,
            GL_VERTEX_ARRAY, GL_TRIANGLE_STRIP, GL_POINTS, glPointSize,
            glDeleteTextures, glActiveTexture, GL_TEXTURE0, GL_TEXTURE1,
            glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
            glGetShaderInfoLog, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
            GL_COMPILE_STATUS, glCreateProgram, glAttachShader, glLinkProgram,
            glGetProgramiv, glGetProgramInfoLog, GL_LINK_STATUS, glUseProgram,
            glGetUniformLocation, glUniform1i, glUniform1f, glUniform3f, glDeleteProgram,
            glDeleteShader,
            glBindAttribLocation, glGetAttribLocation,
            glGenBuffers, glBindBuffer, glBufferData, glDeleteBuffers,
            GL_ARRAY_BUFFER, GL_STATIC_DRAW,
            glEnableVertexAttribArray, glDisableVertexAttribArray, glVertexAttribPointer,
            GL_TRIANGLES, GL_FALSE,
            glTexSubImage2D, glTexSubImage3D,
            # PIP stats overlay (text-as-texture)
            GL_RGBA, GL_UNSIGNED_BYTE,
        )
    except ImportError:
        print("PyOpenGL required:  pip install PyOpenGL PyOpenGL_accelerate", flush=True)
        return

    pygame.init()
    W, H = 1560, 860
    pygame.display.set_caption("Thick Lens — Spectral Cross-Section Viewer")
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)

    # Capture the Pygame/WGL display context handle for GL object sharing,
    # and the HDC for pixel-format matching (prevents err=0 on stricter drivers).
    # Must be called immediately after set_mode() while the context is current.
    _gl_display_hglrc: int = 0
    _gl_display_hdc: int = 0
    _wgl_make_current = None
    if sys.platform == "win32":
        try:
            _wgl_get_current_context = ctypes.windll.opengl32.wglGetCurrentContext
            _wgl_get_current_dc = ctypes.windll.opengl32.wglGetCurrentDC
            _wgl_make_current = ctypes.windll.opengl32.wglMakeCurrent
            _wgl_get_current_context.restype = ctypes.c_void_p
            _wgl_get_current_dc.restype = ctypes.c_void_p
            _wgl_make_current.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            _wgl_make_current.restype = ctypes.c_int
            _gl_display_hglrc = int(_wgl_get_current_context() or 0)
            _gl_display_hdc = int(_wgl_get_current_dc() or 0)
            if _gl_display_hglrc:
                print(f"[gl-share] captured display HGLRC=0x{_gl_display_hglrc:x}"
                      f" HDC=0x{_gl_display_hdc:x}", flush=True)
        except Exception as _hglrc_err:
            print(f"[gl-share] HGLRC capture failed: {_hglrc_err}", flush=True)

    def _restore_display_gl_context() -> None:
        if not (_wgl_make_current and _gl_display_hdc and _gl_display_hglrc):
            return
        try:
            rc = _wgl_make_current(
                ctypes.c_void_p(int(_gl_display_hdc)),
                ctypes.c_void_p(int(_gl_display_hglrc)),
            )
            if not rc:
                print("[gl-share] display context restore failed", flush=True)
        except Exception as _restore_err:
            print(f"[gl-share] display context restore failed: {_restore_err}", flush=True)
    clock = pygame.time.Clock()
    frame_profiler = FrameProfiler(report_every=60) if profile else None

    scene    = SceneConfig()
    scene.image_plate.sensor_res = int(max(4, sensor_res))
    view_h, view_w = H, W
    sidecar  = FreeFrequencySidecar.lazy_prepare(int(DEFAULT_FREQ_HZ.size))
    bench    = ForwardCppLensBench(
        scene=scene,
        freq_hz=DEFAULT_FREQ_HZ.copy(),
        view_h=view_h,
        view_w=view_w,
        sidecar=sidecar,
        field_capture=field_capture,
    )

    # Wire the display HGLRC and HDC to the tracer BEFORE the first submit_rays.
    if _gl_display_hglrc:
        if _gl_display_hdc:
            bench.tracer.set_gl_display_hdc(_gl_display_hdc)
        bench.tracer.set_gl_display_hglrc(_gl_display_hglrc)
        # Pre-upload RGB weights for the GPU blit shader.
        try:
            _wl_nm = np.clip(C_LIGHT / np.maximum(bench.freq_hz, EPS) * 1.0e9, 380.0, 700.0)
            _blit_w = _wavelength_to_rgb_weights(_wl_nm).astype(np.float32)
            bench.tracer.set_uv_blit_weights(_blit_w, mode=0)
            print(f"[gl-share] UV blit weights uploaded ({len(bench.freq_hz)} bands)", flush=True)
        except Exception as _bw_err:
            print(f"[gl-share] blit weight upload failed: {_bw_err}", flush=True)
    bench.sensor_amp_gain      = float(sensor_amp_gain)
    bench.sensor_min_amplitude = float(sensor_min_amplitude)
    bench.emitter_amp_gain     = float(emitter_amp_gain)
    bench.compute_mode         = str(compute_mode)
    print(f"[bench] tris={bench.n_tris}  bands={bench.n_bands}  "
          f"view={view_w}x{view_h}  sensor_amp_gain={bench.sensor_amp_gain}  "
          f"emitter_amp_gain={bench.emitter_amp_gain}  "
          f"sensor_min_amplitude={bench.sensor_min_amplitude}", flush=True)
    # ── Configure C++ sensor image accumulator ─────────────────────────────
    _pip_res = int(max(16, scene.image_plate.sensor_res))
    bench.tracer.configure_sensor_image(
        float(scene.image_plate.x),
        float(scene.image_plate.radius),
        _pip_res,
        0.008,
    )
    if compute_mode in ("gpu", "mixed"):
        try:
            bench.tracer.ensure_pipeline(
                max_children=2,
                seed=13579,
                min_amplitude=float(bench._min_amplitude),
                use_gpu_compute=True,
                gpu_all_stages=(compute_mode == "gpu"),
                shader_dir=_SHADER_DIR,
            )
        finally:
            _restore_display_gl_context()

    def _compile_shader(kind: int, source: str) -> int:
        shader = glCreateShader(kind)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            msg = glGetShaderInfoLog(shader)
            raise RuntimeError(msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg))
        return shader

    vert_src = """
        #version 120
        varying vec2 v_uv;
        void main() {
            v_uv = gl_Vertex.xy * 0.5 + 0.5;
            gl_Position = gl_Vertex;
        }
    """
    frag_src = """
        #version 120
        uniform sampler3D u_field;
        uniform float u_time;
        uniform float u_mode;
        uniform float u_gain;
        varying vec2 v_uv;

        vec3 display_curve(vec3 x) {
            vec3 y = log(vec3(1.0) + max(x, vec3(0.0)));
            return y / (vec3(1.0) + y);
        }

        void main() {
            float t = u_time * 0.16;
            float a = (u_mode < 0.5) ? (0.24 * sin(t)) : ((u_mode < 1.5) ? 0.0 : 1.57079632679);
            float ca = cos(a);
            float sa = sin(a);
            vec2 yz_axis = vec2(ca, sa);
            float x = v_uv.x;
            float v = v_uv.y - 0.5;
            vec2 yz = yz_axis * v;
            vec3 p = vec3(x, yz.x + 0.5, yz.y + 0.5);
            vec3 raw = texture3D(u_field, clamp(p, 0.0, 1.0)).rgb;
            gl_FragColor = vec4(clamp(display_curve(raw * u_gain), 0.0, 1.0), 1.0);
        }
    """
    point_vert_src = """
        #version 120
        uniform float u_time;
        uniform float u_mode;
        uniform float u_gain;
        varying float v_amp;
        varying float v_depth;

        void main() {
            float t = u_time * 0.16;
            float a = (u_mode < 0.5) ? (0.24 * sin(t)) : ((u_mode < 1.5) ? 0.0 : 1.57079632679);
            float ca = cos(a);
            float sa = sin(a);
            vec3 p = gl_Vertex.xyz;
            vec2 yz = p.yz - vec2(0.5);
            float v = dot(yz, vec2(ca, sa));
            gl_Position = vec4(p.x * 2.0 - 1.0, v * 2.0, 0.0, 1.0);
            // Near points (v > 0) get larger, far points smaller — natural depth cue
            gl_PointSize = max(1.0, 1.5 + (v + 0.5) * 2.5);
            v_amp   = gl_Vertex.w * u_gain;
            v_depth = v + 0.5;  // 0 = far, 1 = near
        }
    """
    point_frag_src = """
        #version 120
        uniform float u_class;
        varying float v_amp;
        varying float v_depth;

        vec3 display_curve(vec3 x) {
            vec3 y = log(vec3(1.0) + max(x, vec3(0.0)));
            return y / (vec3(1.0) + y);
        }

        void main() {
            // 0=forward-strike amber  1=forward-volume cyan
            // 2=reverse-volume green  3=reverse-strike magenta
            vec3 base;
            if      (u_class < 0.5) base = vec3(1.00, 0.60, 0.18);
            else if (u_class < 1.5) base = vec3(0.18, 0.88, 1.00);
            else if (u_class < 2.5) base = vec3(0.18, 1.00, 0.45);
            else                    base = vec3(1.00, 0.28, 0.72);
            vec3 c = display_curve(vec3(v_amp)) * base;
            // Additive alpha: near = bright/opaque, far = dim (shows through volume)
            float alpha = clamp(0.2 + 0.8 * v_depth, 0.0, 1.0);
            gl_FragColor = vec4(clamp(c, 0.0, 1.0), alpha);
        }
    """
    vs = _compile_shader(GL_VERTEX_SHADER, vert_src)
    fs = _compile_shader(GL_FRAGMENT_SHADER, frag_src)
    volume_prog = glCreateProgram()
    glAttachShader(volume_prog, vs)
    glAttachShader(volume_prog, fs)
    glLinkProgram(volume_prog)
    if not glGetProgramiv(volume_prog, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(volume_prog)
        raise RuntimeError(msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg))
    pvs = _compile_shader(GL_VERTEX_SHADER, point_vert_src)
    pfs = _compile_shader(GL_FRAGMENT_SHADER, point_frag_src)
    point_prog = glCreateProgram()
    glAttachShader(point_prog, pvs)
    glAttachShader(point_prog, pfs)
    glLinkProgram(point_prog)
    if not glGetProgramiv(point_prog, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(point_prog)
        raise RuntimeError(msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg))

    # ── UV-mesh shader: renders every scene triangle lit by the UV-splat atlas ──
    # Uses the same Y/Z projection rotation as the volume + point shaders so
    # mesh, volume, and scatter are always co-registered in screen space.
    mesh_vert_src = """
        #version 130
        attribute vec3 a_pos;
        attribute vec2 a_uv;
        attribute float a_layer;
        uniform float u_time;
        uniform float u_mode;
        varying vec2 v_uv;
        varying float v_layer;
        void main() {
            float t = u_time * 0.16;
            float a = (u_mode < 0.5) ? (0.24 * sin(t))
                    : ((u_mode < 1.5) ? 0.0 : 1.57079632679);
            float ca = cos(a); float sa = sin(a);
            vec2 yz = a_pos.yz - vec2(0.5);
            float vp = dot(yz, vec2(ca, sa));
            gl_Position = vec4(a_pos.x * 2.0 - 1.0, vp * 2.0, 0.0, 1.0);
            v_uv = a_uv;
            v_layer = a_layer;
        }
    """
    mesh_frag_src = """
        #version 130
        uniform sampler2DArray u_uv_pages;
        uniform float u_gain;
        varying vec2 v_uv;
        varying float v_layer;

        vec3 display_curve(vec3 x) {
            vec3 y = log(vec3(1.0) + max(x, vec3(0.0)));
            return y / (vec3(1.0) + y);
        }

        void main() {
            vec3 tex = max(texture(u_uv_pages, vec3(v_uv, v_layer)).rgb, vec3(0.0));
            vec3 c   = clamp(display_curve(tex * u_gain), 0.0, 1.0);
            float alpha = clamp(max(max(c.r, c.g), c.b), 0.0, 1.0);
            if (alpha < 0.005) discard;
            gl_FragColor = vec4(c, alpha);
        }
    """
    mvs = _compile_shader(GL_VERTEX_SHADER,   mesh_vert_src)
    mfs = _compile_shader(GL_FRAGMENT_SHADER, mesh_frag_src)
    mesh_prog = glCreateProgram()
    glAttachShader(mesh_prog, mvs)
    glAttachShader(mesh_prog, mfs)
    glBindAttribLocation(mesh_prog, 0, "a_pos")
    glBindAttribLocation(mesh_prog, 1, "a_uv")
    glBindAttribLocation(mesh_prog, 2, "a_layer")
    glLinkProgram(mesh_prog)
    if not glGetProgramiv(mesh_prog, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(mesh_prog)
        raise RuntimeError(msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg))
    _mesh_loc_pos = glGetAttribLocation(mesh_prog, "a_pos")
    _mesh_loc_uv  = glGetAttribLocation(mesh_prog, "a_uv")
    _mesh_loc_layer = glGetAttribLocation(mesh_prog, "a_layer")

    # ── GL volume textures ───────────────────────────────────────── #
    tex_field = glGenTextures(1)
    blank_vol = np.zeros((bench._field_nz, bench._field_ny, bench._field_nx, 3), dtype=np.float32)
    _empty4 = np.zeros((0, 4), dtype=np.float32)
    # List is mutated in-place so draw_surface_points closure always sees current arrays
    surface_verts_by_class: List[np.ndarray] = [_empty4, _empty4, _empty4, _empty4]
    _svc = surface_verts_by_class  # alias used by the closure
    fullscreen_quad = np.ascontiguousarray(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    glBindTexture(GL_TEXTURE_3D, tex_field)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, bench._field_nx, bench._field_ny, bench._field_nz,
                 0, GL_RGB, GL_FLOAT, blank_vol)

    # ── PIP 2-D texture (reads back C++ sensor accumulator) ─────────────────
    pip_vert_src = """
        #version 120
        varying vec2 v_uv;
        void main() {
            v_uv        = gl_Vertex.xy * 0.5 + 0.5;
            gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
        }
    """
    pip_frag_src = """
        #version 120
        uniform sampler2D u_pip;
        uniform float u_border;          /* 0=sensor image, 1=HUD text texture */
        uniform vec3  u_border_col;      /* border colour when u_border>0.5     */
        varying vec2 v_uv;
        void main() {
            if (u_border > 0.5) {
                /* HUD mode: blit u_pip as-is (pre-rendered text rgba) */
                gl_FragColor = texture2D(u_pip, v_uv);
                return;
            }
            vec3 c   = texture2D(u_pip, v_uv).rgb;
            vec2 d   = abs(v_uv - vec2(0.5)) * 2.0;
            float edge = max(d.x, d.y);
            /* Soft inner fade near very edge */
            float brd  = 1.0 - smoothstep(0.90, 1.0, edge);
            /* Hard 3-px cyan border band */
            float in_b = step(0.970, edge);
            vec3  col  = mix(c, u_border_col, in_b);
            float alph = max(brd * 0.92, in_b);
            gl_FragColor = vec4(col, alph);
        }
    """
    pip_vs   = _compile_shader(GL_VERTEX_SHADER,   pip_vert_src)
    pip_fs   = _compile_shader(GL_FRAGMENT_SHADER, pip_frag_src)
    pip_prog = glCreateProgram()
    glAttachShader(pip_prog, pip_vs)
    glAttachShader(pip_prog, pip_fs)
    glLinkProgram(pip_prog)
    if not glGetProgramiv(pip_prog, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(pip_prog)
        raise RuntimeError(msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg))

    tex_pip = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_pip)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    _pip_blank = np.zeros((_pip_res, _pip_res, 3), dtype=np.float32)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _pip_res, _pip_res, 0,
                 GL_RGB, GL_FLOAT, _pip_blank)

    # ── Per-physical-group analytical UV page array ─────────────────────────
    tex_uv_pages = glGenTextures(1)
    _uv_layers = max(1, len(bench.uv_page_bank.groups) if bench.uv_page_bank is not None else 1)
    _uv_res = int(bench.uv_page_bank.res if bench.uv_page_bank is not None else UV_PAGE_RES_DEFAULT)
    _uv_blank = np.zeros((_uv_layers, _uv_res, _uv_res, 4), dtype=np.float32)
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_uv_pages)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA16F, _uv_res, _uv_res, _uv_layers,
                 0, GL_RGBA, GL_FLOAT, _uv_blank)

    # ── BaseGLRenderer: bridge tex_uv_pages into the base-material shader ───
    # The renderer and shaders already exist; this wires the shared texture ID
    # so uEmitUv samples from the same array that the blit shader writes into.
    _gl_renderer = BaseGLRenderer(bench.material_db)
    _gl_renderer.init_gl()
    _gl_renderer.set_emit_uv_texture_id(int(tex_uv_pages))
    _scene_vao: list = [None]  # lazily built once uv_page_bank groups are ready

    def _build_scene_vao() -> None:
        bank = bench.uv_page_bank
        if bank is None or not bank.groups:
            return
        n_tris = bench.n_tris
        tv = bench.tri_vertices  # (N, 3, 3) float64 xyz
        # Normalise into [0, 1]³ — same mapping as _ensure_mesh_vbo.
        x0  = float(scene.x_min);  x1  = float(scene.x_max)
        r_v = float(scene.view_radius)
        xsp = max(1e-8, x1 - x0);  yzsp = max(1e-8, 2.0 * r_v)
        pos_n = np.empty((n_tris, 3, 3), dtype=np.float32)
        pos_n[..., 0] = np.clip((tv[..., 0] - x0)    / xsp,  0.0, 1.0)
        pos_n[..., 1] = np.clip((tv[..., 1] + r_v)   / yzsp, 0.0, 1.0)
        pos_n[..., 2] = np.clip((tv[..., 2] + r_v)   / yzsp, 0.0, 1.0)
        pos_flat = pos_n.reshape(-1, 3)                          # (N*3, 3)
        # Flat normals.
        e1 = (tv[:, 1] - tv[:, 0]).astype(np.float32)
        e2 = (tv[:, 2] - tv[:, 0]).astype(np.float32)
        raw_n = np.cross(e1, e2)
        nlen  = np.linalg.norm(raw_n, axis=1, keepdims=True)
        norms = (raw_n / np.where(nlen > 1e-12, nlen, 1.0)).astype(np.float32)
        norm_flat = np.repeat(norms, 3, axis=0)                  # (N*3, 3)
        uv_flat = np.zeros((n_tris * 3, 2), dtype=np.float32)
        # Use the actual scene material IDs so that emissive surfaces (red probe
        # sphere, light sources) carry their real emission_rgb into the fragment
        # shader.  The renderer's derive_emissive_area_lights will pick those up
        # and handle illumination — no manual light setup needed here.
        mat_flat = np.repeat(bench.tri_mat_ids, 3)               # (N*3,) ints
        # Group IDs for self-exclusion in the fragment shader.
        gid_flat = np.zeros(n_tris * 3, dtype=np.int32)
        for g in bank.groups:
            if g.tri_ids.size <= 0:
                continue
            if g.group_id < 0:
                continue
            for tid in g.tri_ids:
                base = int(tid) * 3
                gid_flat[base:base + 3] = int(g.group_id)
        mat_v  = np.ascontiguousarray(mat_flat, dtype=np.int32)
        gid_v  = np.ascontiguousarray(gid_flat, dtype=np.int32)
        # Build verts8: [x, y, z, nx, ny, nz, u, v]
        verts8 = np.concatenate([pos_flat, norm_flat, uv_flat], axis=1)
        verts8 = np.ascontiguousarray(verts8, dtype=np.float32)
        # All surfaces double-sided so the fly camera can view geometry from
        # any angle.  The red probe and sensor plate are always double-sided
        # for field-side visibility as well.
        cull_v = np.ones(n_tris * 3, dtype=np.int32)
        from OpenGL.GL import (
            glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
            glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
            glVertexAttribIPointer,
            GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_INT,
        )
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        mbo = glGenBuffers(1)
        gbo = glGenBuffers(1)
        cbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts8.nbytes, verts8, GL_STATIC_DRAW)
        stride = 8 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 2, GL_FLOAT, False, stride, ctypes.c_void_p(24))
        glBindBuffer(GL_ARRAY_BUFFER, mbo)
        glBufferData(GL_ARRAY_BUFFER, mat_v.nbytes, mat_v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribIPointer(2, 1, GL_INT, 4, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, gbo)
        glBufferData(GL_ARRAY_BUFFER, gid_v.nbytes, gid_v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(4)
        glVertexAttribIPointer(4, 1, GL_INT, 4, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, cbo)
        glBufferData(GL_ARRAY_BUFFER, cull_v.nbytes, cull_v, GL_STATIC_DRAW)
        glEnableVertexAttribArray(5)
        glVertexAttribIPointer(5, 1, GL_INT, 4, ctypes.c_void_p(0))
        glBindVertexArray(0)
        _scene_vao[0] = (int(vao), n_tris * 3, int(vbo), int(mbo), int(gbo), int(cbo))
        # Let the renderer derive lights from all emissive surfaces in the scene
        # (red probe sphere, source triangles, etc.) — no manual light setup.
        _gl_renderer.derive_emissive_area_lights(verts8, mat_v, gid_v, min_emitter_group_id=-999)
        print(f"[gl-renderer] scene VAO built: {n_tris*3} verts, "
              f"{len(bank.groups)} UV groups", flush=True)

    # HUD text texture — RGBA8, sized to a reasonable stats strip width × height
    _HUD_W, _HUD_H = 256, 48
    tex_hud = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_hud)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    _hud_blank = np.zeros((_HUD_H, _HUD_W, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _HUD_W, _HUD_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, _hud_blank)

    # PIP viewport: square centred horizontally, near top of window.
    _pip_dim = int(min(W * 0.22, H * 0.22))
    _pip_vx  = (W - _pip_dim) // 2
    _pip_vy  = 6

    # Lazy font for PIP stats overlay – created on first draw to avoid init cost.
    _pip_font: list = [None]  # mutable container so closure can write it
    _u_border      = glGetUniformLocation(pip_prog, "u_border")
    _u_border_col  = glGetUniformLocation(pip_prog, "u_border_col")

    def _draw_quad_with_pip_prog(tex: int, vx: int, vy: int, vw: int, vh: int,
                                 hud_mode: bool = False) -> None:
        """Draw a fullscreen quad in the given viewport using pip_prog."""
        glViewport(vx, vy, vw, vh)
        glUseProgram(pip_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(pip_prog, "u_pip"), 0)
        glUniform1f(_u_border, 1.0 if hud_mode else 0.0)
        glUniform3f(_u_border_col, 0.35, 0.85, 1.0)   # cyan border
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE if not hud_mode else GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, fullscreen_quad)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_BLEND)
        glUseProgram(0)

    def draw_pip() -> None:
        img = bench.tracer.get_sensor_image()   # (res, res, 3) float32 [0,1]
        has_img = img.shape[0] > 0

        if has_img:
            glBindTexture(GL_TEXTURE_2D, tex_pip)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.shape[1], img.shape[0],
                            GL_RGB, GL_FLOAT, img)  # C++ already emits rows bottom-first

        # Always draw the pip quad — border is baked into the shader.
        # When sensor has no data yet the quad draws a pure cyan border on
        # transparent background (additive over the scene), which is visible.
        _draw_quad_with_pip_prog(tex_pip, _pip_vx, _pip_vy, _pip_dim, _pip_dim)

        # --- BDPT stats text rendered as a texture quad above the PIP ---
        try:
            stats = bench.tracer.get_bdpt_stats()
        except Exception:
            stats = None
        if stats is not None:
            if _pip_font[0] is None:
                pygame.font.init()
                _pip_font[0] = (pygame.font.SysFont("consolas", 11) or
                                pygame.font.SysFont("monospace", 11))
            font = _pip_font[0]
            nd = stats.get("nearest_dist_m", -1.0)
            bc = stats.get("best_collinearity", 0.0)
            es = stats.get("exact_snaps", 0)
            nm = stats.get("near_miss_count", 0)
            dist_str = (f"dyz {nd*1000:.2f}mm" if nd >= 0.0 else "dyz --")
            hud_surf = pygame.Surface((_HUD_W, _HUD_H), pygame.SRCALPHA)
            hud_surf.fill((0, 0, 0, 0))
            line_h = font.get_linesize()
            for i, txt in enumerate([dist_str,
                                      f"col {bc:.3f}",
                                      f"snaps {es} nr {nm}"]):
                rendered = font.render(txt, True, (80, 210, 255))
                hud_surf.blit(rendered, (2, i * (line_h + 1)))
            # pygame surface → RGBA8 numpy array (y-flipped for GL)
            raw = pygame.surfarray.array_alpha(
                hud_surf.convert_alpha())  # shape (W, H)
            rgb_raw = pygame.surfarray.array3d(hud_surf)  # shape (W, H, 3)
            rgba = np.zeros((_HUD_H, _HUD_W, 4), dtype=np.uint8)
            rgba[:, :, :3] = np.transpose(rgb_raw, (1, 0, 2))
            rgba[:, :, 3]  = np.transpose(raw,     (1, 0))
            rgba = rgba[::-1].copy()  # flip rows: pygame y-down → GL y-up
            glBindTexture(GL_TEXTURE_2D, tex_hud)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _HUD_W, _HUD_H, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            # Place stats strip just above the PIP top edge
            hud_vx = _pip_vx
            hud_vy = _pip_vy + _pip_dim + 2
            hud_vw = _HUD_W
            hud_vh = _HUD_H
            _draw_quad_with_pip_prog(tex_hud, hud_vx, hud_vy, hud_vw, hud_vh,
                                     hud_mode=True)


    _vol_tex_size = [bench._field_nx, bench._field_ny, bench._field_nz]  # already allocated at init

    def upload_volume(tex, rgb_zyx3: np.ndarray) -> None:
        arr = np.ascontiguousarray(rgb_zyx3, dtype=np.float32)
        w, h, d = arr.shape[2], arr.shape[1], arr.shape[0]
        glBindTexture(GL_TEXTURE_3D, tex)
        if w != _vol_tex_size[0] or h != _vol_tex_size[1] or d != _vol_tex_size[2]:
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F, w, h, d, 0, GL_RGB, GL_FLOAT, arr)
            _vol_tex_size[0], _vol_tex_size[1], _vol_tex_size[2] = w, h, d
        else:
            glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, d, GL_RGB, GL_FLOAT, arr)

    def draw_volume(vp_x: int, vp_y: int, vp_w: int, vp_h: int, mode: int, gain: float, t_now: float) -> None:
        glViewport(vp_x, vp_y, vp_w, vp_h)
        glUseProgram(volume_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, tex_field)
        glUniform1i(glGetUniformLocation(volume_prog, "u_field"), 0)
        glUniform1f(glGetUniformLocation(volume_prog, "u_time"), float(t_now))
        glUniform1f(glGetUniformLocation(volume_prog, "u_mode"), float(mode))
        glUniform1f(glGetUniformLocation(volume_prog, "u_gain"), float(gain))
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, fullscreen_quad)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableClientState(GL_VERTEX_ARRAY)
        glUseProgram(0)

    # Display-class colours in same order as shader: amber, cyan, green, magenta
    _CLASS_NAMES = ["fwd-strike", "fwd-volume", "rev-volume", "rev-strike"]

    def draw_surface_points(mode: int, gain: float, t_now: float) -> None:
        any_pts = any(v.shape[0] > 0 for v in _svc)
        if not any_pts:
            return
        glUseProgram(point_prog)
        glUniform1f(glGetUniformLocation(point_prog, "u_time"), float(t_now))
        glUniform1f(glGetUniformLocation(point_prog, "u_mode"), float(mode))
        glUniform1f(glGetUniformLocation(point_prog, "u_gain"), float(gain))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # additive — depth order doesn't matter
        glEnableClientState(GL_VERTEX_ARRAY)
        for cls, verts in enumerate(_svc):
            if verts.shape[0] == 0:
                continue
            glUniform1f(glGetUniformLocation(point_prog, "u_class"), float(cls))
            glVertexPointer(4, GL_FLOAT, 0, verts)
            glDrawArrays(GL_POINTS, 0, int(verts.shape[0]))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_BLEND)
        glUseProgram(0)

    # ── UV-mesh draw: physical groups render as layers in a texture array ───
    _mesh_vbo: list = [None]   # [vbo_pos, vbo_uv, vbo_layer] once uploaded
    _mesh_n_verts: list = [0]
    _uv_last_update_s: list = [-1.0]
    _uv_mode: list = ["combined"]
    _uv_shared_tex_id: list = [0]  # [int tex ID] when C++ GPU blit is active, else 0

    def _ensure_mesh_vbo() -> bool:
        if _mesh_vbo[0] is not None:
            return True
        bank = bench.uv_page_bank
        if bank is None or not bank.groups:
            return False
        pos_parts = []
        uv_parts = []
        layer_parts = []
        v_all = bench.tri_vertices
        x0  = float(scene.x_min);  x1  = float(scene.x_max)
        r_v = float(scene.view_radius)
        xsp = max(1e-8, x1 - x0);  yzsp = max(1e-8, 2.0 * r_v)
        for g in bank.groups:
            if g.tri_ids.size <= 0:
                continue
            v = v_all[g.tri_ids]
            pos = np.empty((v.shape[0], 3, 3), dtype=np.float32)
            pos[..., 0] = np.clip((v[..., 0] - x0) / xsp,    0.0, 1.0)
            pos[..., 1] = np.clip((v[..., 1] + r_v) / yzsp,  0.0, 1.0)
            pos[..., 2] = np.clip((v[..., 2] + r_v) / yzsp,  0.0, 1.0)
            pos_parts.append(pos.reshape(-1, 3))
            uv_parts.append(g.uv_coords.reshape(-1, 2))
            layer_parts.append(np.full((g.tri_ids.size * 3,), float(g.layer), dtype=np.float32))
        if not pos_parts:
            return False
        pos_flat = np.ascontiguousarray(np.concatenate(pos_parts, axis=0), dtype=np.float32)
        uv_flat = np.ascontiguousarray(np.concatenate(uv_parts, axis=0), dtype=np.float32)
        layer_flat = np.ascontiguousarray(np.concatenate(layer_parts, axis=0), dtype=np.float32)
        vbos = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[0])
        glBufferData(GL_ARRAY_BUFFER, pos_flat.nbytes, pos_flat, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glBufferData(GL_ARRAY_BUFFER, uv_flat.nbytes, uv_flat, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glBufferData(GL_ARRAY_BUFFER, layer_flat.nbytes, layer_flat, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        _mesh_vbo[0]     = vbos
        _mesh_n_verts[0] = int(pos_flat.shape[0])
        return True

    def draw_uv_mesh(gain: float, t_now: float) -> None:
        if not _ensure_mesh_vbo():
            return
        bank = bench.uv_page_bank
        if bank is None:
            return

        # Lazily build the BaseGLRenderer VAO from the UV page bank geometry.
        if _scene_vao[0] is None:
            _build_scene_vao()

        # Lazily discover the shared GPU texture ID from the C++ blit shader.
        if _uv_shared_tex_id[0] == 0:
            try:
                tid = bench.tracer.get_uv_pages_tex_id()
                if tid:
                    _uv_shared_tex_id[0] = int(tid)
                    _gl_renderer.set_emit_uv_texture_id(int(tid))
                    print(f"[gl-share] GPU-direct UV tex id={tid} active", flush=True)
            except Exception:
                pass

        if _uv_shared_tex_id[0]:
            # GPU-direct path: C++ blit shader already wrote RGBA16F into the
            # shared texture.  No CPU round-trip needed; just bind and draw.
            active_tex = _uv_shared_tex_id[0]
        else:
            # CPU fallback path: readback → NumPy → glTexSubImage3D.
            if t_now - _uv_last_update_s[0] >= 0.05:
                pages = bank.update_from_tracer(bench.tracer, bench.freq_hz, mode=_uv_mode[0])
                glBindTexture(GL_TEXTURE_2D_ARRAY, tex_uv_pages)
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                                int(pages.shape[2]), int(pages.shape[1]), int(pages.shape[0]),
                                GL_RGBA, GL_FLOAT, np.ascontiguousarray(pages, dtype=np.float32))
                _uv_last_update_s[0] = float(t_now)
            active_tex = tex_uv_pages
        glUseProgram(mesh_prog)
        glUniform1f(glGetUniformLocation(mesh_prog, "u_time"), float(t_now))
        glUniform1f(glGetUniformLocation(mesh_prog, "u_mode"), 0.0)
        glUniform1f(glGetUniformLocation(mesh_prog, "u_gain"), float(gain))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, active_tex)
        glUniform1i(glGetUniformLocation(mesh_prog, "u_uv_pages"), 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        vbos = _mesh_vbo[0]
        glBindBuffer(GL_ARRAY_BUFFER, vbos[0])
        glEnableVertexAttribArray(_mesh_loc_pos)
        glVertexAttribPointer(_mesh_loc_pos, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[1])
        glEnableVertexAttribArray(_mesh_loc_uv)
        glVertexAttribPointer(_mesh_loc_uv,  2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vbos[2])
        glEnableVertexAttribArray(_mesh_loc_layer)
        glVertexAttribPointer(_mesh_loc_layer, 1, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDrawArrays(GL_TRIANGLES, 0, _mesh_n_verts[0])
        glDisableVertexAttribArray(_mesh_loc_pos)
        glDisableVertexAttribArray(_mesh_loc_uv)
        glDisableVertexAttribArray(_mesh_loc_layer)
        glDisable(GL_BLEND)
        glUseProgram(0)

        # Drive BaseGLRenderer: fly-camera perspective when fly_mode is active,
        # otherwise fall back to the orthographic auto-wiggle.
        if _scene_vao[0] is not None:
            if fly_mode:
                # ── Perspective MVP ───────────────────────────────────────── #
                _aspect  = float(W) / float(H)
                _cp = math.cos(fly_pitch); _sp = math.sin(fly_pitch)
                _cy = math.cos(fly_yaw);   _sy = math.sin(fly_yaw)
                _fwd_d   = np.array([_cp*_sy, _sp, _cp*_cy], dtype=np.float64)
                _target  = fly_pos + _fwd_d
                _world_up = np.array([0.0, 1.0, 0.0])   # Y is up
                if abs(float(np.dot(_fly_norm(_fwd_d), _world_up))) > 0.97:
                    _world_up = np.array([0.0, 0.0, 1.0])
                V = _fly_lookat(fly_pos, _target, _world_up)
                P = _fly_persp(math.radians(60.0), _aspect, 0.005, 5.0)
                _mvp_fly = (P @ V).astype(np.float32)
                mvp_col  = np.ascontiguousarray(_mvp_fly.T.ravel(), dtype=np.float32)
                mv_col   = np.ascontiguousarray(V.T.ravel(), dtype=np.float32)
            else:
                t   = t_now * 0.16
                a   = 0.24 * math.sin(t)
                ca  = math.cos(a);  sa = math.sin(a)
                mvp = np.array([
                    [ 2,     0,     0, -1          ],
                    [ 0,  2*ca,  2*sa, -(ca + sa)  ],
                    [ 0,     0,     0,  0          ],
                    [ 0,     0,     0,  1          ],
                ], dtype=np.float32)
                mvp_col = np.ascontiguousarray(mvp.T.ravel(), dtype=np.float32)
                mv_col  = np.ascontiguousarray(np.eye(4, dtype=np.float32).ravel())
            vao_id, n_verts = _scene_vao[0][0], _scene_vao[0][1]
            _gl_renderer.draw_mesh(vao_id, n_verts, mvp_col, mv_col)

    # ── Clip plane helpers ───────────────────────────────────────── #
    r         = float(scene.view_radius)
    clip_step = r * 0.02
    clip_top  = r    # start: show full scene; Q cuts inward
    clip_side = r

    def apply_clip(rgb: np.ndarray, clip_val: float) -> np.ndarray:
        """Zero image rows above clip_val in world space.

        Row 0 = +view_radius world Y/Z (top of image).
        row_clip = first visible row (world coord <= clip_val).
        """
        H_img   = rgb.shape[0]
        row_cut = int((0.5 - clip_val / (2.0 * r)) * H_img)
        row_cut = max(0, min(H_img, row_cut))
        if row_cut == 0:
            return rgb
        out              = rgb.copy()
        out[:row_cut, :] = 0.0
        return out

    # ── Background trace ─────────────────────────────────────────── #
    executor   = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    trace_fut: Optional[concurrent.futures.Future] = None

    field_gain      = 0.6
    field_leak      = 0.0
    max_bounces     = 16
    rays_per_emitter = 8
    seed            = 13579
    frame           = 0
    _display_frame  = 0   # independent counter for display-update rate limiting
    paused          = False
    closing         = threading.Event()

    projection_axis = 1  # 1 = X/Y top projection, 2 = X/Z side projection, 0 = subtle auto-wiggle blend
    auto_wiggle = True
    volume_gain = 1.0

    # ── Fly camera (F to toggle) ──────────────────────────────────────────── #
    # Positions are in the same [0,1]³ normalised space the VAO uses.
    # Scene centre = (0.5, 0.5, 0.5).  Start outside the scene on the -Z side,
    # looking toward the centre of the optical bench.
    fly_mode  = False
    fly_pos   = np.array([0.5, 0.5, -0.4], dtype=np.float64)
    fly_yaw   = 0.0              # yaw=0 → forward = +Z (into the scene)
    fly_pitch = 0.0
    _FLY_SPEED = 0.6   # normalised units / second
    _FLY_SENS  = 0.20  # degrees per pixel of mouse movement

    # KPN back-pressure gate: block submit until C++ intent queue drains to below
    # capacity.  display_pipeline_records() still runs every frame so the viewer
    # stays live while the pipeline is catching up.
    _MAX_IN_FLIGHT = 65_536

    def _trace(rpe: int, sd: int, mb: int):
        import time as _t
        target_in_flight = 0 if ray_mode == "backward" else _MAX_IN_FLIGHT
        while not closing.is_set() and bench.tracer.in_flight_count() > target_in_flight:
            _t.sleep(0.002)   # back off; let drain-loop consume Q_intent
        if closing.is_set():
            return
        if ray_mode != "backward":
            bench.trace_forward(rpe, sd, max_bounces=mb)
        if ray_mode != "forward":
            bench.trace_sensor_cast(rpe, sd ^ 0x5A5A, max_bounces=mb)

    try:
        while True:
            if frame_profiler is not None:
                frame_profiler.begin("events")
            for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.MOUSEMOTION and fly_mode:
                        dx, dy = ev.rel
                        fly_yaw   += math.radians(dx * _FLY_SENS)
                        fly_pitch  = float(np.clip(
                            fly_pitch - math.radians(dy * _FLY_SENS),
                            -math.pi * 0.44, math.pi * 0.44,
                        ))
                    if ev.type == pygame.KEYDOWN:
                        k = ev.key
                        if k == pygame.K_ESCAPE:
                            if fly_mode:
                                fly_mode = False
                                pygame.mouse.set_visible(True)
                                pygame.event.set_grab(False)
                            else:
                                return
                        elif k == pygame.K_f:
                            fly_mode = not fly_mode
                            if fly_mode:
                                pygame.mouse.set_visible(False)
                                pygame.event.set_grab(True)
                            else:
                                pygame.mouse.set_visible(True)
                                pygame.event.set_grab(False)
                        elif k == pygame.K_SPACE and not fly_mode:
                            paused = not paused
                        elif not fly_mode:
                            if k == pygame.K_x:
                                projection_axis = 0
                                auto_wiggle = True
                            elif k == pygame.K_y:
                                projection_axis = 1
                                auto_wiggle = False
                            elif k == pygame.K_z:
                                projection_axis = 2
                                auto_wiggle = False
                            elif k == pygame.K_q:
                                clip_top  = max(-r, clip_top  - clip_step)
                            elif k == pygame.K_e:
                                clip_top  = min( r, clip_top  + clip_step)
                            elif k == pygame.K_c:
                                clip_side = min( r, clip_side + clip_step)
                            elif k == pygame.K_r:
                                clip_top = clip_side = r
                            elif k == pygame.K_9:
                                field_gain = max(0.0, field_gain - 0.05)
                            elif k == pygame.K_0:
                                field_gain = min(5.0, field_gain + 0.05)
                            elif k == pygame.K_u:
                                modes = ["combined", "forward", "sensor", "difference"]
                                _uv_mode[0] = modes[(modes.index(_uv_mode[0]) + 1) % len(modes)]
                                _uv_last_update_s[0] = -1.0
                                print(f"[uv-mode] {_uv_mode[0]}", flush=True)
                            elif k == pygame.K_LEFTBRACKET:
                                bench.intent_shuffle = max(0.0, bench.intent_shuffle - 0.1)
                                bench.tracer.set_intent_shuffle(float(bench.intent_shuffle))
                                print(f"[shuffle] {bench.intent_shuffle:.2f}", flush=True)
                            elif k == pygame.K_RIGHTBRACKET:
                                bench.intent_shuffle = min(1.0, bench.intent_shuffle + 0.1)
                                bench.tracer.set_intent_shuffle(float(bench.intent_shuffle))
                                print(f"[shuffle] {bench.intent_shuffle:.2f}", flush=True)
            if frame_profiler is not None:
                frame_profiler.end("events")

            # ── Fly-camera WASD movement (main loop, not in draw closure) ── #
            if fly_mode:
                _dt_s     = max(0.001, clock.get_time() * 1e-3)
                _keys_now = pygame.key.get_pressed()
                _cp = math.cos(fly_pitch); _sp = math.sin(fly_pitch)
                _cy = math.cos(fly_yaw);   _sy = math.sin(fly_yaw)
                _fwd   = np.array([_cp*_sy, _sp, _cp*_cy], dtype=np.float64)
                _right = _fly_norm(np.cross(np.array([0.0, 1.0, 0.0]), _fwd))
                if np.linalg.norm(_right) < 1e-9:
                    _right = np.array([1.0, 0.0, 0.0])
                _up_v  = _fly_norm(np.cross(_fwd, _right))
                _spd   = _FLY_SPEED * _dt_s
                if _keys_now[pygame.K_LSHIFT] or _keys_now[pygame.K_RSHIFT]:
                    _spd *= 3.0
                if _keys_now[pygame.K_w]: fly_pos = fly_pos + _fwd   * _spd
                if _keys_now[pygame.K_s]: fly_pos = fly_pos - _fwd   * _spd
                if _keys_now[pygame.K_a]: fly_pos = fly_pos - _right * _spd
                if _keys_now[pygame.K_d]: fly_pos = fly_pos + _right * _spd
                if _keys_now[pygame.K_q]: fly_pos = fly_pos + _up_v  * _spd
                if _keys_now[pygame.K_e]: fly_pos = fly_pos - _up_v  * _spd

            if not paused and trace_fut is None:
                trace_fut = executor.submit(
                    _trace, rays_per_emitter, (seed + frame) & 0x7FFFFFFF,
                    max_bounces)
                frame += 1

            if trace_fut is not None and trace_fut.done():
                try:
                    trace_fut.result()
                except Exception as exc:
                    print(f"[trace] {exc}", flush=True)
                trace_fut = None

            # Refresh field + UV textures every frame for continuous trickle display.
            _display_frame += 1
            if True:
                try:
                    if frame_profiler is not None:
                        frame_profiler.begin("display_records")
                    field_texels, new_vbc = bench.display_pipeline_records(field_gain=field_gain)
                    if frame_profiler is not None:
                        frame_profiler.end("display_records")
                        frame_profiler.begin("upload_volume")
                    upload_volume(tex_field, field_texels)
                    _gl_renderer.set_field_volume_texture(int(tex_field), field_gain)
                    if frame_profiler is not None:
                        frame_profiler.end("upload_volume")
                    for _ci in range(4):
                        _svc[_ci] = new_vbc[_ci]
                except Exception as exc:
                    if frame_profiler is not None:
                        frame_profiler.end("display_records")
                        frame_profiler.end("upload_volume")
                    print(f"[display] {exc}", flush=True)

            if auto_wiggle or projection_axis == 0:
                shader_mode = 0
                axis_label = "AUTO X/Y↔X/Z"
            elif projection_axis == 2:
                shader_mode = 2
                axis_label = "X/Z"
            else:
                shader_mode = 1
                axis_label = "X/Y"

            pygame.display.set_caption(
                f"Thick Lens — {axis_label} | bounces={max_bounces} | "
                f"shuffle={bench.intent_shuffle:.2f} ([/] to adjust)"
            )

            glClearColor(8/255, 8/255, 10/255, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glViewport(0, 0, W, H)
            t_now = pygame.time.get_ticks() * 0.001
            if frame_profiler is not None:
                frame_profiler.begin("draw_uv_mesh")
            draw_uv_mesh(volume_gain, t_now)
            if frame_profiler is not None:
                frame_profiler.end("draw_uv_mesh")
                frame_profiler.begin("draw_pip")
            try:
                draw_pip()
            except Exception as _pip_exc:
                print(f"[pip] {_pip_exc}", flush=True)
            if frame_profiler is not None:
                frame_profiler.end("draw_pip")
                frame_profiler.begin("flip")
            pygame.display.flip()
            if frame_profiler is not None:
                frame_profiler.end("flip")
                try:
                    ps = bench.tracer.pipeline_stats()
                    extra = (
                        f"inflight={int(ps.get('in_flight', 0))} "
                        f"qout={int(ps.get('output_queue_depth', 0))} "
                        f"uv_rb={float(ps.get('gpu_uv_readback_mb', 0.0)):.1f}MB/"
                        f"{int(ps.get('gpu_uv_readback_count', 0))} "
                        f"hit_rb={float(ps.get('gpu_hit_readback_mb', 0.0)):.1f}MB/"
                        f"{int(ps.get('gpu_hit_readback_count', 0))}"
                    )
                except Exception:
                    extra = ""
                frame_profiler.tick(extra)
            clock.tick(60)
    finally:
        import gc
        closing.set()
        # 1. Stop the drain loop first so it releases the pipeline before we destroy it.
        bench._drain_stop.set()
        if bench._drain_thread is not None:
            bench._drain_thread.join(timeout=2.0)
        # 2. Wait for any pending trace future so its submit_rays/pipeline call finishes.
        if trace_fut is not None:
            trace_fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        # 3. Force immediate destruction of the C++ pipeline and its worker threads.
        #    PyRayTracer.__del__ calls ray_pipeline_destroy() which joins all std::threads.
        del bench
        gc.collect()
        glDeleteTextures(1, [tex_field])
        glDeleteTextures(1, [tex_pip])
        glDeleteTextures(1, [tex_hud])
        glDeleteTextures(1, [tex_uv_pages])
        if _mesh_vbo[0] is not None:
            glDeleteBuffers(3, _mesh_vbo[0])
        if _scene_vao[0] is not None:
            vao_id, _, vbo, mbo, gbo, cbo = _scene_vao[0]
            from OpenGL.GL import glDeleteVertexArrays
            glDeleteVertexArrays(1, [vao_id])
            glDeleteBuffers(4, [vbo, mbo, gbo, cbo])
        glDeleteProgram(volume_prog)
        glDeleteProgram(point_prog)
        glDeleteProgram(pip_prog)
        glDeleteProgram(mesh_prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        glDeleteShader(pvs)
        glDeleteShader(pfs)
        glDeleteShader(pip_vs)
        glDeleteShader(pip_fs)
        glDeleteShader(mvs)
        glDeleteShader(mfs)
        pygame.quit()


def run_uv_smoke(
    ray_mode: str = "bdpt",
    sensor_res: int = 32,
    sensor_amp_gain: float = 1.0,
    sensor_min_amplitude: float = 0.0,
    compute_mode: str = "cpu",
    steps: int = 2,
    timeout_s: float = 12.0,
) -> int:
    """Headless end-to-end UV wiring check using the real ray pipeline."""
    import gc
    import time

    scene = SceneConfig()
    scene.image_plate.sensor_res = int(max(4, sensor_res))
    sidecar = FreeFrequencySidecar.lazy_prepare(int(DEFAULT_FREQ_HZ.size))
    bench = ForwardCppLensBench(
        scene=scene,
        freq_hz=DEFAULT_FREQ_HZ.copy(),
        view_h=360,
        view_w=640,
        sidecar=sidecar,
    )
    bench.compute_mode = str(compute_mode)
    bench.sensor_amp_gain = float(sensor_amp_gain)
    bench.sensor_min_amplitude = float(sensor_min_amplitude)
    bench.tracer.configure_sensor_image(
        float(scene.image_plate.x),
        float(scene.image_plate.radius),
        int(max(16, scene.image_plate.sensor_res)),
        0.008,
    )

    rc = 1
    try:
        print(
            "[uv-smoke-start]",
            f"mode={ray_mode}",
            f"compute={compute_mode}",
            f"sensor_res={scene.image_plate.sensor_res}",
            f"steps={int(max(1, steps))}",
            flush=True,
        )
        for step in range(int(max(1, steps))):
            seed = 20260516 + step * 101
            if ray_mode != "backward":
                bench.trace_forward(1, seed, max_bounces=1)
            if ray_mode != "forward":
                bench.trace_sensor_cast(1, seed ^ 0x5A5A, max_bounces=1)

            t0 = time.perf_counter()
            t_last_probe = 0.0
            while True:
                try:
                    in_flight = int(bench.tracer.in_flight_count())
                except Exception:
                    in_flight = 0
                now = time.perf_counter()
                if bench.uv_page_bank is not None and now - t_last_probe >= 0.5:
                    bench.uv_page_bank.refresh_summaries(bench.tracer)
                    t_last_probe = now
                    if bench.uv_page_bank.any_hot_data():
                        break
                if in_flight <= 0:
                    break
                if now - t0 > float(timeout_s):
                    print(f"[uv-smoke-timeout] step={step} in_flight={in_flight}", flush=True)
                    break
                time.sleep(0.02)

            if bench.uv_page_bank is not None:
                bench.uv_page_bank.update_from_tracer(bench.tracer, bench.freq_hz, mode="combined")

        any_uv = False
        if bench.uv_page_bank is not None:
            for g in bench.uv_page_bank.groups:
                s = dict(g.last_summary)
                nz = int(s.get("nonzero_texels", 0) or 0)
                fwd = float(s.get("total_forward", 0.0) or 0.0)
                sen = float(s.get("total_sensor", 0.0) or 0.0)
                if g.hot and (nz > 0 or fwd > 0.0 or sen > 0.0):
                    any_uv = True
                print(
                    "[uv-smoke-group]",
                    f"name={g.name}",
                    f"hot={int(g.hot)}",
                    f"gid={int(g.group_id)}",
                    f"layer={int(g.layer)}",
                    f"tris={int(g.tri_ids.size)}",
                    f"nonzero={nz}",
                    f"forward={fwd:.6g}",
                    f"sensor={sen:.6g}",
                    flush=True,
                )
        rc = 0 if any_uv else 2
        print(f"[uv-smoke-result] rc={rc}", flush=True)
        return rc
    finally:
        bench._drain_stop.set()
        if bench._drain_thread is not None:
            bench._drain_thread.join(timeout=2.0)
        del bench
        gc.collect()


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Spectral lens bench")
    _ap.add_argument(
        "--ray-mode",
        choices=["bdpt", "forward", "backward"],
        default="bdpt",
        help="bdpt = forward+backward (default); forward = forward rays only; backward = sensor-cast rays only",
    )
    _ap.add_argument(
        "--sensor-res",
        type=int,
        default=64,
        metavar="N",
        help="Sensor pixel-grid side length (default: 64); active pixels about pi/4*N^2",
    )
    _ap.add_argument(
        "--sensor-gain",
        type=float,
        default=1.0,
        metavar="G",
        help="Multiply sensor ray launch amplitude by G to pre-compensate Fresnel dropoff (default: 1.0)",
    )
    _ap.add_argument(
        "--emitter-gain",
        type=float,
        default=1.0,
        metavar="G",
        help="Multiply forward emitter ray launch amplitude by G (default: 1.0)",
    )
    _ap.add_argument(
        "--sensor-min-amp",
        type=float,
        default=0.0,
        metavar="A",
        help="Amplitude floor for sensor rays; 0.0 = never kill (single-photon mode, default)",
    )
    _ap.add_argument(
        "--compute-mode",
        choices=["cpu", "gpu", "mixed"],
        default="gpu",
        help=(
            "cpu  = CPU workers only, no GPU compute shaders; "
            "gpu  = GPU handles all pipeline stages (default); "
            "mixed = GPU compute enabled but CPU workers also active (compete for work)"
        ),
    )
    _ap.add_argument(
        "--uv-smoke-exit",
        action="store_true",
        help="Run a headless end-to-end UV pipeline smoke test and exit",
    )
    _ap.add_argument(
        "--uv-smoke-steps",
        type=int,
        default=2,
        metavar="N",
        help="Number of trace/readback iterations for --uv-smoke-exit",
    )
    _ap.add_argument(
        "--profile",
        action="store_true",
        help="Print rolling Python frame timings and pipeline readback counters",
    )
    _ap.add_argument(
        "--no-field",
        action="store_true",
        help="Disable volumetric field capture (saves ~128×64×64 complex grid memory per frame)",
    )
    _args = _ap.parse_args()
    if _args.uv_smoke_exit:
        raise SystemExit(run_uv_smoke(
            ray_mode=_args.ray_mode,
            sensor_res=_args.sensor_res,
            sensor_amp_gain=_args.sensor_gain,
            sensor_min_amplitude=_args.sensor_min_amp,
            compute_mode=_args.compute_mode,
            steps=_args.uv_smoke_steps,
        ))
    run(
        ray_mode=_args.ray_mode,
        sensor_res=_args.sensor_res,
        sensor_amp_gain=_args.sensor_gain,
        sensor_min_amplitude=_args.sensor_min_amp,
        emitter_amp_gain=_args.emitter_gain,
        compute_mode=_args.compute_mode,
        profile=bool(_args.profile),
        field_capture=not _args.no_field,
    )
