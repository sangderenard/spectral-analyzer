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

from camera_designer.compound_optics import (
    ApertureStop as CompoundApertureStop,
    CompoundLens,
    ConicSurface as CompoundConicSurface,
    RayBundle,
    TerminationReason,
)
from camera_designer.lens_assembly import LensAssemblySpec

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
BDPT_PIPELINE_CAP = 25_000_000
DEFAULT_FREQ_HZ = (C_LIGHT / np.linspace(700e-9, 380e-9, MAX_SPECTRAL_BANDS)).astype(np.float64)
DEBUG_BDPT_BACKTRACE_ONLY_DEFAULT = False
# When False (the default), backward-sensor sub-path records (PIXEL_CONE /
# APERTURE_PUPIL) are NOT deposited into the physical 3-D field grid because
# they are unresolved half-paths.  Set True only to restore legacy behaviour
# where all endpoint records were written to the field regardless of stream
# origin.
ENABLE_UNSAFE_BACKWARD_FIELD_DEPOSIT: bool = False
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


def _cull_finite_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    *coarrays,
    label: str = "",
) -> tuple:
    """Return a tuple of (origins, directions, *coarrays) with non-finite rows removed.

    A row is invalid when any element of origins OR directions is NaN or ±inf.
    All coarrays must have the same first-axis length as origins.
    Returns the original arrays unchanged (same objects) when all rows are finite.
    """
    valid = np.all(np.isfinite(origins), axis=1) & np.all(np.isfinite(directions), axis=1)
    if np.all(valid):
        return (origins, directions) + coarrays
    n_bad = int(np.count_nonzero(~valid))
    print(
        f"[cull-infinite{':' + label if label else ''}] removed {n_bad}/{len(valid)} rays with non-finite coords",
        flush=True,
    )
    return (origins[valid], directions[valid]) + tuple(a[valid] for a in coarrays)


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
    x: float = 1.25
    radius: float = 0.040    # 120 6×6 format: 56mm square frame, half-diagonal ≈ 39.6mm
    pixels: int = 64          # mesh tessellation rings (polar disc)
    sensor_res: int = 64      # pixel-grid side length; ~π/4·res² sites active in disc
    bokeh_rays: int = 4       # stencil rays fired per pixel site per call
    bokeh_stencil_frac: float = 0.25  # stencil radius = this fraction of aperture radius
    shutter_mode: str = "open"        # open | closed | iris | sliding_x | sliding_y
    shutter_open: float = 1.0         # 0..1 opening fraction
    shutter_center_u: float = 0.5     # sliding/iris centre in film UV
    shutter_center_v: float = 0.5
    shutter_softness: float = 0.0     # UV transition width; 0 = hard edge


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
class DiffuserWaveTubeSpec:
    """Geometry for one diffuser disc, ready for WaveTube.register()."""
    entry_tri_ids:      np.ndarray   # c1 face (tube-interior side) int32
    exit_tri_ids:       np.ndarray   # c0 face (stage side) int32
    entry_pos:          np.ndarray   # centre of entry face (3,) metres
    exit_pos:           np.ndarray   # centre of exit face  (3,) metres
    axis:               np.ndarray   # unit vec from entry to exit
    tube_radius_m:      float
    diffuser_thickness: float
    ior_real:           float = 1.45
    ior_imag:           float = 0.002


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
    wave_spec_out: Optional[List["DiffuserWaveTubeSpec"]] = None,
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
        _diff_entry_ids: List[int] = [] if wave_spec_out is not None else None  # type: ignore[assignment]
        _diff_exit_ids:  List[int] = [] if wave_spec_out is not None else None  # type: ignore[assignment]
        _build_thin_disc_element_oriented(
            center=np.ascontiguousarray(diff_center, dtype=np.float64),
            normal=np.ascontiguousarray(axis, dtype=np.float64),
            radius=diff_r,
            thickness=diff_thick,
            tri_list=tri_list,
            mat_ids=mat_ids,
            mat_idx=idx_diffuser,
            n_theta=96,
            entry_tri_ids=_diff_entry_ids,
            exit_tri_ids=_diff_exit_ids,
        )
        if wave_spec_out is not None:
            wave_spec_out.append(DiffuserWaveTubeSpec(
                entry_tri_ids      = np.ascontiguousarray(_diff_entry_ids, dtype=np.int32),
                exit_tri_ids       = np.ascontiguousarray(_diff_exit_ids,  dtype=np.int32),
                entry_pos          = np.ascontiguousarray(diff_center + diff_half * axis, dtype=np.float64),
                exit_pos           = np.ascontiguousarray(diff_center - diff_half * axis, dtype=np.float64),
                axis               = np.ascontiguousarray(-axis, dtype=np.float64),
                tube_radius_m      = diff_r,
                diffuser_thickness = diff_thick,
            ))

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


def _default_optical_design_spec():
    from camera_software.optical_design import OpticalDesignSpec
    return OpticalDesignSpec(
        focal_length_range_m=(0.075, 0.100),  # 120 6×6: ~80mm normal FOV
        zoom=0.40,
        focus_distance_m=1.0,
        f_number=2.8,
        entrance_x_m=1.08,
        sensor_x_m=1.25,
        sensor_clearance_m=0.030,
        image_radius_m=0.040,                 # matches ImagePlateConfig.radius
        min_air_gap_m=0.008,
        group_thickness_m=0.018,
        max_group_radius_m=0.075,             # medium format needs larger elements
    )


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
    baffle2_x: float = 1.18  # Just before the default sensor plane.
    baffle2_aperture: float = 0.1
    screen_x: float = 1.25
    screen_radius: float = 0.040    # matches image_plate.radius for 120 6×6 format
    view_radius: float = 0.24
    auto_fit_view: bool = True
    view_aspect: float = 1560.0 / 860.0
    enable_debug_plate: bool = False
    debug_plate_x: float = 0.50
    debug_plate_radius: float = 0.20
    object_plane: ObjectPlaneConfig = field(default_factory=ObjectPlaneConfig)
    image_plate: ImagePlateConfig = field(default_factory=ImagePlateConfig)
    include_legacy_stage: bool = False
    subject_scene_mode: str = "orbiters"
    subject_time_s: float = 0.0
    subject_scale: float = 0.12
    subject_depth_scale: Optional[float] = None
    subject_x: Optional[float] = None
    subject_y: float = 0.0
    subject_z: float = 0.0
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
    # Ring light: emissive annular ring mounted flush with the front of the lens barrel.
    # The forward-facing face is emissive (ring_light_emission); the rear face is black
    # matte so light only exits toward the subject.
    ring_light_enabled: bool = True
    ring_light_width_m: float = 0.018      # radial width of the emitting annulus (m)
    ring_light_emission: float = 50.0      # emission scale (relative to source material)
    ring_light_n_sectors: int = 72         # angular tessellation segments
    # Exit pupil / field stop aperture between last lens and sensor. When enabled
    # this limits light transmission post-optics, letting you tune spectral content
    # and transmission efficiency independently of the entrance aperture.
    exit_pupil_x: float = 1.85
    exit_pupil_radius: float = 0.008   # 8 mm radius
    exit_pupil_thickness: float = 0.0003  # 0.3 mm — blade aperture
    lens_hood_front_radius: float = 0.115
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
    # Fallback 4-group design for 120 6×6 format (~80mm EFL).
    # The optical design solver replaces this at runtime; these are
    # only used when the solver is disabled or fails.
    lens_stack: List[LensConfig] = field(default_factory=lambda: [
        LensConfig(center_x=0.98, thickness=0.046, aperture_radius=0.052, radius_front=0.110, radius_back=0.135, ior=1.52),
        LensConfig(center_x=1.08, thickness=0.034, aperture_radius=0.044, radius_front=0.180, radius_back=0.180, ior=1.62),
        LensConfig(center_x=1.19, thickness=0.046, aperture_radius=0.048, radius_front=0.130, radius_back=0.115, ior=1.52),
        LensConfig(center_x=1.34, thickness=0.050, aperture_radius=0.052, radius_front=0.150, radius_back=0.150, ior=1.57),
    ])
    optical_design: Optional[object] = field(default_factory=_default_optical_design_spec)
    aperture_model: str = "geometry"  # geometry | wave3d


def _scene_lenses(scene: SceneConfig) -> List[LensConfig]:
    design = getattr(scene, "optical_design", None)
    if design is not None:
        try:
            solved = design
            if hasattr(design, "form") and not hasattr(design, "groups"):
                import dataclasses as _dc
                from camera_software.optical_design import solve_four_group_zoom_surrogate

                # ── Physical camera placement from scene geometry ──────────────
                # G1 entrance: placed so scene.object_plane.x is exactly at the
                # design focus distance.
                # Sensor: placed at paraxial image distance past the minimum group
                # span.  This keeps the camera compact (correct tube length for
                # the focal length) and ensures the image actually forms at the
                # sensor plane instead of hundreds of mm past it.
                obj_x   = float(scene.object_plane.x)
                f_m     = float(design.target_focal_length_m)
                u_m     = float(max(design.focus_distance_m, f_m * 1.05))

                # Optional macro supplementary: adjust effective focus distance.
                macro_cfg = getattr(scene, "macro_lens", None)
                if macro_cfg is not None:
                    # Thin lens: 1/u_eff = 1/u + P_macro
                    _tc = float(getattr(macro_cfg, "thickness", 0.0))
                    _rf = float(getattr(macro_cfg, "radius_front", 0.0))
                    _rb = float(getattr(macro_cfg, "radius_back",  0.0))
                    _n  = float(getattr(macro_cfg, "ior", 1.52))
                    if abs(_rf) > 1e-6:
                        _p_macro = (_n - 1.0) * (1.0/abs(_rf) + 1.0/abs(_rb) if abs(_rb) > 1e-6 else 1.0/abs(_rf))
                        _inv = 1.0/u_m + _p_macro
                        if _inv > 1e-6:
                            u_m = 1.0 / _inv

                # Thin-lens paraxial image distance at this focus.
                _denom = 1.0/f_m - 1.0/u_m
                v_m = (1.0/_denom) if abs(_denom) > 1e-9 else f_m * 20.0
                # Clamp sensor to scene bounds.
                v_m = min(v_m, float(scene.x_max) - float(obj_x) - u_m - 0.05)

                entrance_x = obj_x + u_m
                min_group_span = (
                    float(design.group_count - 1) * float(design.min_air_gap_m)
                    + float(design.group_count)   * float(design.group_thickness_m)
                )
                sensor_x = entrance_x + min_group_span + v_m + float(design.sensor_clearance_m)

                # The camera barrel can be wider than the scene/stage bore, but
                # scene.tube_radius is a scene scale.  Keep it stable so camera
                # placement does not stretch the rest of the world.
                camera_barrel_r = float(design.max_group_radius_m + 0.012)

                design = _dc.replace(
                    design,
                    entrance_x_m=round(entrance_x, 4),
                    sensor_x_m=round(sensor_x, 4),
                )
                scene.image_plate.x     = round(sensor_x, 4)
                scene.screen_x          = round(sensor_x, 4)
                print(
                    "[camera-placement]",
                    f"obj_x={obj_x:.4f}",
                    f"u={u_m*1e3:.1f}mm  v={v_m*1e3:.1f}mm",
                    f"entrance={entrance_x:.4f}  sensor={sensor_x:.4f}",
                    f"camera_barrel_r={camera_barrel_r*1e3:.1f}mm",
                    f"scene_tube_r={float(scene.tube_radius)*1e3:.1f}mm",
                    flush=True,
                )

                solved = solve_four_group_zoom_surrogate(design)
            if hasattr(solved, "apply_to_scene"):
                scene.optical_design = solved
                solved.apply_to_scene(scene)
                print(
                    "[optical-design]",
                    f"form={getattr(getattr(solved, 'spec', None), 'form', 'unknown')}",
                    f"groups={len(getattr(solved, 'groups', []))}",
                    f"f_eff={float(getattr(solved, 'effective_focal_length_m', 0.0))*1e3:.1f}mm",
                    f"sensor_error={float(getattr(solved, 'sensor_error_m', 0.0))*1e3:.2f}mm",
                    flush=True,
                )
                groups = getattr(solved, "groups", ())
                if len(groups) >= 3 and getattr(scene, "iris_aperture", None) is None:
                    g2 = groups[1]
                    g3 = groups[2]
                    ap_x = float(0.5 * (float(g2.x_m) + float(g3.x_m)))
                    ap_r = float(max(1.0e-4, getattr(solved.spec, "aperture_radius_m", 0.018)))
                    ap_outer = float(min(float(scene.tube_radius), ap_r * 1.55))
                    scene.iris_aperture = IrisApertureConfig(
                        enabled=True,
                        x_pos=round(ap_x, 6),
                        r_inner=round(ap_r, 6),
                        r_outer=round(max(ap_r * 1.4, ap_outer), 6),
                        n_blades=6,
                    )
                    print(
                        "[aperture-placed]",
                        f"x={ap_x:.4f}  (between G2@{float(g2.x_m):.4f} G3@{float(g3.x_m):.4f})",
                        f"r_inner={ap_r*1e3:.2f}mm",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[optical-design] solve/apply failed: {exc}", flush=True)
            scene.optical_design = None
    lenses = list(getattr(scene, "lens_stack", []) or [])
    if not lenses:
        lenses = [scene.lens]

    # Optional macro supplementary group: positive close-up lens prepended at
    # the front of the stack.  It reduces the effective focal length, allowing
    # the main lens to focus closer than its minimum native focus distance.
    macro_cfg = getattr(scene, "macro_lens", None)
    if macro_cfg is not None and lenses:
        # Place macro element just in front of G1 with a small air gap.
        g1_front = float(lenses[0].x_front)
        gap = float(getattr(scene, "min_air_gap_m", 0.006) if hasattr(scene, "min_air_gap_m") else 0.006)
        thick = float(getattr(macro_cfg, "thickness", 0.012))
        macro_center = g1_front - gap - thick * 0.5
        import dataclasses as _dc
        placed_macro = _dc.replace(macro_cfg, center_x=round(macro_center, 5))
        lenses = [placed_macro] + lenses
        print(
            "[macro-lens]",
            f"center_x={macro_center:.4f}",
            f"thickness={thick*1e3:.1f}mm",
            f"radius_front={getattr(placed_macro,'radius_front',0.0)*1e3:.1f}mm",
            flush=True,
        )

    return lenses


def _auto_fit_scene_view_to_mesh(scene: SceneConfig, tri_arr: np.ndarray) -> None:
    """Fit orthographic and field-grid bounds to the generated physical mesh."""
    if not bool(getattr(scene, "auto_fit_view", True)):
        return
    arr = np.asarray(tri_arr, dtype=np.float64)
    if arr.size <= 0:
        return
    pts = arr.reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite):
        return
    pts = pts[finite]
    x0 = float(np.min(pts[:, 0]))
    x1 = float(np.max(pts[:, 0]))
    r = float(np.max(np.abs(pts[:, 1:3]))) if pts.shape[1] >= 3 else float(scene.view_radius)

    semantic_x = [
        float(getattr(scene, "source_x", x0)),
        float(getattr(getattr(scene, "object_plane", None), "x", x0)),
        float(getattr(scene, "exit_pupil_x", x1)),
        float(getattr(getattr(scene, "image_plate", None), "x", x1)),
    ]
    x0 = min(x0, min(semantic_x))
    x1 = max(x1, max(semantic_x))
    r = max(
        r,
        float(getattr(scene, "source_radius", 0.0)),
        float(getattr(getattr(scene, "object_plane", None), "radius", 0.0)),
        float(getattr(getattr(scene, "image_plate", None), "radius", 0.0)),
    )

    x_content_span = max(0.10, x1 - x0)
    x_pad = max(0.025, 0.035 * x_content_span)
    y_content_span = max(0.10, 2.0 * (r * 1.08 + 0.010))
    aspect = float(max(1.0e-6, getattr(scene, "view_aspect", 1.0)))
    x_span = max(x_content_span + 2.0 * x_pad, y_content_span * aspect)
    y_span = max(y_content_span, x_span / aspect)
    x_mid = 0.5 * (x0 + x1)
    old = (float(scene.x_min), float(scene.x_max), float(scene.view_radius))
    scene.x_min = float(x_mid - 0.5 * x_span)
    scene.x_max = float(x_mid + 0.5 * x_span)
    scene.view_radius = float(0.5 * y_span)
    new = (float(scene.x_min), float(scene.x_max), float(scene.view_radius))
    if any(abs(a - b) > 1.0e-6 for a, b in zip(old, new)):
        print(
            "[view-fit]",
            f"x=({scene.x_min:.4f},{scene.x_max:.4f})",
            f"radius={scene.view_radius:.4f}",
            flush=True,
        )


def _compound_lens_from_scene(scene: SceneConfig) -> CompoundLens:
    """Build the canonical parametric optical model from SceneConfig.

    Mesh generation and optical transport both consume the same LensConfig list,
    so changing the scene's ordinary lens parameters changes the visible mesh and
    the parametric transfer chain together.
    """
    lens = CompoundLens()
    elements = []
    n_air = 1.0
    for cfg in _scene_lenses(scene):
        if not _lens_is_valid(cfg):
            continue
        n_glass = float(max(1.0, cfg.ior))
        elements.append((
            float(cfg.x_front),
            CompoundConicSurface(
                x_pos=float(cfg.x_front),
                R_curvature=float(cfg.radius_front),
                n_before=n_air,
                n_after=n_glass,
                aperture_r=float(cfg.aperture_radius),
                conic_k=0.0,
            ),
        ))
        elements.append((
            float(cfg.x_back),
            CompoundConicSurface(
                x_pos=float(cfg.x_back),
                R_curvature=-float(cfg.radius_back),
                n_before=n_glass,
                n_after=n_air,
                aperture_r=float(cfg.aperture_radius),
                conic_k=0.0,
            ),
        ))

    iris = getattr(scene, "iris_aperture", None)
    if iris is not None and bool(getattr(iris, "enabled", False)):
        elements.append((
            float(iris.x_pos),
            CompoundApertureStop(
                x_pos=float(iris.x_pos),
                r_clear=float(iris.r_inner),
                n_medium=n_air,
            ),
        ))

    for _x, element in sorted(elements, key=lambda item: item[0]):
        lens.add(element)
    return lens


def _compound_lens_from_stack(lens_stack, iris_aperture=None) -> CompoundLens:
    """Build CompoundLens directly from a lens stack without calling _scene_lenses().

    Used for live element adjustment so apply_to_scene() is never triggered.
    """
    lens = CompoundLens()
    elements = []
    n_air = 1.0
    for cfg in lens_stack:
        if not _lens_is_valid(cfg):
            continue
        n_glass = float(max(1.0, cfg.ior))
        elements.append((float(cfg.x_front), CompoundConicSurface(
            x_pos=float(cfg.x_front),
            R_curvature=float(cfg.radius_front),
            n_before=n_air, n_after=n_glass,
            aperture_r=float(cfg.aperture_radius), conic_k=0.0,
        )))
        elements.append((float(cfg.x_back), CompoundConicSurface(
            x_pos=float(cfg.x_back),
            R_curvature=-float(cfg.radius_back),
            n_before=n_glass, n_after=n_air,
            aperture_r=float(cfg.aperture_radius), conic_k=0.0,
        )))
    if iris_aperture is not None and bool(getattr(iris_aperture, "enabled", False)):
        elements.append((float(iris_aperture.x_pos), CompoundApertureStop(
            x_pos=float(iris_aperture.x_pos),
            r_clear=float(iris_aperture.r_inner),
            n_medium=n_air,
        )))
    for _x, element in sorted(elements, key=lambda item: item[0]):
        lens.add(element)
    return lens


def _paraxial_image_x(compound_lens: "CompoundLens", object_x: float) -> float:
    """Paraxial image x for an on-axis point at *object_x*.

    Uses the CompoundLens system matrix.  A marginal ray at h=0 is propagated
    from the object plane to the first refractive surface, then through the
    full system, then in air until it crosses the axis.

    Returns float('inf') when the system is afocal or divergent for this object.
    """
    elements = getattr(compound_lens, "_elements", [])
    x_first = None
    x_last  = None
    for el in elements:
        if isinstance(el, (CompoundConicSurface, )):
            if x_first is None:
                x_first = float(el.x_pos)
            x_last = float(el.x_pos)
    if x_first is None or x_last is None:
        return float("inf")

    d = max(0.0, x_first - float(object_x))  # object-to-first-surface propagation
    M = compound_lens._paraxial_matrix()      # system matrix first→last surface

    # Propagate from object to first surface: state [h=0, nu=1] → [d, 1]
    # Then through system: M @ [d, 1]
    A, B = float(M[0, 0]), float(M[0, 1])
    C, D = float(M[1, 0]), float(M[1, 1])
    Bt = A * d + B
    Dt = C * d + D

    # Image distance from last surface: v = −Bt/Dt
    if abs(Dt) < 1.0e-12:
        return float("inf")
    v = -Bt / Dt
    return float(x_last + v)


def _solver_focal_plane_x(scene) -> float:
    """Focal plane from the optical design solver's thin-lens group model.

    Uses exactly the same computation the solver used when placing the sensor, so
    the result is within sensor_error (~0.02mm) of scene.image_plate.x.
    Falls back to float('inf') when the solved design is unavailable.
    """
    design = getattr(scene, "optical_design", None)
    if design is None:
        return float("inf")
    groups = getattr(design, "groups", None)
    if not groups:
        return float("inf")
    try:
        from camera_software.optical_design import image_distance_for_object as _img_dist_fn
        obj_x  = float(scene.object_plane.x)
        img_d  = _img_dist_fn(groups, obj_x)
        if not math.isfinite(img_d):
            return float("inf")
        last_x = float(max(g.x_m for g in groups))
        return float(last_x + img_d)
    except Exception:
        return float("inf")


def _probe_focus_coc(
    compound_lens: "CompoundLens",
    object_x: float,
    sensor_x: float,
    aperture_r: float,
    *,
    n_rings: int = 4,
    n_phi: int = 12,
    n_sweep: int = 32,
    sweep_half_range: float = 0.015,
) -> dict:
    """Trace a polar fan of rays to measure the circle of confusion at the sensor
    and locate the minimum-spot-size plane (true ray-traced focus).

    Python-side exact algebraic trace via CompoundLens.evaluate_bundle() — no GPU,
    runs in ~1 ms for the default 49-ray fan.

    Returns a dict:
        coc_at_sensor_mm : RMS transverse spot radius (mm) at sensor_x
        min_coc_x        : X position (m) of the tightest spot
        min_coc_mm       : tightest RMS spot radius (mm)
        n_passed         : rays that cleared the assembly
    """
    # Aim rays at the first conic surface (entrance of the lens)
    elements = getattr(compound_lens, "_elements", [])
    x_first = None
    for el in elements:
        if isinstance(el, CompoundConicSurface):
            x_first = float(el.x_pos)
            break
    if x_first is None:
        x_first = float(sensor_x) - 0.050

    obj_x  = float(object_x)
    snsr_x = float(sensor_x)
    ap_r   = float(aperture_r)

    origins_list: list = []
    dirs_list:    list = []
    for ir in range(n_rings):
        r   = ap_r * (ir + 1) / n_rings
        n_a = n_phi if ir > 0 else 1
        for ia in range(n_a):
            phi = 2.0 * math.pi * ia / n_a
            hy  = r * math.cos(phi)
            hz  = r * math.sin(phi)
            origins_list.append([obj_x, 0.0, 0.0])
            d = np.array([x_first - obj_x, hy, hz], dtype=np.float64)
            d /= max(1.0e-15, float(np.linalg.norm(d)))
            dirs_list.append(d.tolist())

    bundle = RayBundle(
        np.array(origins_list, dtype=np.float64),
        np.array(dirs_list,    dtype=np.float64),
    )
    result      = compound_lens.evaluate_bundle(bundle)
    passed_mask = result.status == int(TerminationReason.PASSED.value)
    n_passed    = int(np.sum(passed_mask))
    _nan        = float("nan")
    if n_passed < 2:
        return {"coc_at_sensor_mm": _nan, "min_coc_x": _nan, "min_coc_mm": _nan, "n_passed": n_passed}

    out_o = result.origins[passed_mask]     # (n_passed, 3) exit position at last surface
    out_d = result.directions[passed_mask]  # (n_passed, 3) exit direction

    def _rms_spot_at(x_plane: float) -> float:
        fwd  = out_d[:, 0] > 1.0e-10
        if np.sum(fwd) < 2:
            return _nan
        oo = out_o[fwd];  od = out_d[fwd]
        t  = (x_plane - oo[:, 0]) / od[:, 0]
        ok = t >= 0.0
        if np.sum(ok) < 2:
            return _nan
        oo = oo[ok];  od = od[ok];  t = t[ok]
        hy = oo[:, 1] + t * od[:, 1]
        hz = oo[:, 2] + t * od[:, 2]
        return float(np.sqrt(np.mean(hy * hy + hz * hz)))

    coc_sensor = _rms_spot_at(snsr_x)

    # Sweep around the paraxial estimate to find minimum CoC
    try:
        px = _paraxial_image_x(compound_lens, obj_x)
    except Exception:
        px = snsr_x
    if not math.isfinite(px):
        px = snsr_x

    x_sweep  = np.linspace(px - sweep_half_range, px + sweep_half_range, n_sweep)
    coc_vals = np.array([_rms_spot_at(float(x)) for x in x_sweep])
    valid    = np.isfinite(coc_vals)
    if not np.any(valid):
        return {
            "coc_at_sensor_mm": coc_sensor * 1000.0 if math.isfinite(coc_sensor) else _nan,
            "min_coc_x": px,
            "min_coc_mm": _nan,
            "n_passed": n_passed,
        }

    best_i  = int(np.argmin(coc_vals[valid]))
    min_x   = float(x_sweep[valid][best_i])
    min_coc = float(coc_vals[valid][best_i])

    return {
        "coc_at_sensor_mm": coc_sensor * 1000.0 if math.isfinite(coc_sensor) else _nan,
        "min_coc_x": min_x,
        "min_coc_mm": min_coc * 1000.0,
        "n_passed": n_passed,
    }


def _import_subject_scene(
    scene: SceneConfig,
    db: MaterialDatabase,
    tris: List[np.ndarray],
    mats: List[int],
    source_tri_ids: List[int],
    object_tri_ids: List[int],
) -> None:
    """Import the reusable orbiter/saddle demo as the subject in front of the camera."""
    mode = str(getattr(scene, "subject_scene_mode", "") or "").strip()
    if not mode or mode.lower() in ("none", "off"):
        return
    try:
        import test_basic_gl_cpp_window as subject_mod
    except Exception as exc:
        print(f"[subject-scene] import failed: {exc}", flush=True)
        return

    subject_db, subject_idx = subject_mod.register_materials()
    verts8, _mat_per_v, _gid_per_v, mat_per_tri, _groups = subject_mod.scene_for_phase(
        subject_idx,
        float(getattr(scene, "subject_time_s", 0.0)),
        scene_mode=mode,
    )
    pts = np.asarray(verts8[:, 0:3], dtype=np.float64).reshape(-1, 3, 3)
    mat_per_tri = np.asarray(mat_per_tri, dtype=np.int32).reshape(-1)
    center = np.asarray(getattr(subject_mod, "SCENE_CENTER", np.zeros(3)), dtype=np.float64).reshape(3)
    scale = float(max(1.0e-6, getattr(scene, "subject_scale", 0.12)))
    depth_override = getattr(scene, "subject_depth_scale", None)
    depth_scale = scale if depth_override is None else float(max(0.0, depth_override))
    sx = getattr(scene, "subject_x", None)
    subject_x = float(scene.object_plane.x if sx is None else sx)
    subject_y = float(getattr(scene, "subject_y", 0.0))
    subject_z = float(getattr(scene, "subject_z", 0.0))

    old_to_new: Dict[int, int] = {}
    emissive_old: set[int] = set()
    for name in getattr(subject_db, "_order", []):
        old_i = int(subject_idx.get(name, -1))
        if old_i < 0:
            continue
        material = subject_db._materials[name]
        new_i = db.register(f"subject_{name}", material)
        old_to_new[old_i] = int(new_i)
        emission = np.asarray(
            material.get("emission_rgb", [0.0, 0.0, 0.0]) if isinstance(material, dict) else getattr(material, "emission_rgb", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if emission.size >= 3 and float(np.max(np.abs(emission[:3]))) > 1.0e-8:
            emissive_old.add(old_i)

    start = len(tris)
    for i, tri in enumerate(pts):
        p = np.empty_like(tri, dtype=np.float64)
        # Original demo camera looks down -Z.  For the thick-lens demo it is a
        # camera subject: original Z becomes optical depth, while original X/Y
        # remain transverse subject coordinates.
        p[:, 0] = subject_x + (tri[:, 2] - center[2]) * depth_scale
        p[:, 1] = subject_y + (tri[:, 1] - center[1]) * scale
        # Negate X→Z so original-camera-right (+X) maps to thick-lens-camera-right (−Z).
        # Without the negation, orbiters render behind the centre globe in the side view.
        p[:, 2] = subject_z - (tri[:, 0] - center[0]) * scale
        old_mat = int(mat_per_tri[i]) if i < mat_per_tri.size else -1
        mat_idx = old_to_new.get(old_mat, 0)
        tri_id = len(tris)
        tris.append(np.ascontiguousarray(p, dtype=np.float64))
        mats.append(int(mat_idx))
        object_tri_ids.append(tri_id)
        if old_mat in emissive_old:
            source_tri_ids.append(tri_id)

    print(
        "[subject-scene]",
        f"mode={mode}",
        f"tris={len(tris) - start}",
        f"emitters={len(source_tri_ids)}",
        f"scale={scale:.4f}",
        f"depth_scale={depth_scale:.4f}",
        f"center_x={subject_x:.4f}",
        flush=True,
    )


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
    entry_tri_ids: Optional[List[int]] = None,
    exit_tri_ids:  Optional[List[int]] = None,
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
        idx_c0 = len(tri_list)
        # c0 = center - h*n: stage-side face, normal must point away from disc toward -n.
        # Reversed winding flips the normal from +n to -n so front-face is visible from outside.
        _append_tri(tri_list, mat_ids, c0, ring0[j], ring0[i], mat_idx)
        idx_c1 = len(tri_list)
        # c1 = center + h*n: tube-interior-side face, normal must point away from disc toward +n.
        # Reversed winding flips the normal from -n to +n so front-face is visible from tube interior.
        _append_tri(tri_list, mat_ids, c1, ring1[i], ring1[j], mat_idx)
        # Side walls: outward-pointing radial normals are correct as-is.
        _append_tri(tri_list, mat_ids, ring0[i], ring1[i], ring1[j], mat_idx)
        _append_tri(tri_list, mat_ids, ring0[i], ring1[j], ring0[j], mat_idx)
        if exit_tri_ids  is not None: exit_tri_ids.append(idx_c0)
        if entry_tri_ids is not None: entry_tri_ids.append(idx_c1)


def _lens_front_x(lens: LensConfig, r: np.ndarray) -> np.ndarray:
    R = float(lens.radius_front)
    c = lens.x_front + R
    s = 1.0 if R >= 0.0 else -1.0
    return c - s * np.sqrt(np.maximum(0.0, R * R - r * r))


def _lens_back_x(lens: LensConfig, r: np.ndarray) -> np.ndarray:
    R = float(lens.radius_back)
    c = lens.x_back - R
    s = 1.0 if R >= 0.0 else -1.0
    return c + s * np.sqrt(np.maximum(0.0, R * R - r * r))


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


def _build_ring_light(
    x_pos: float,
    r_inner: float,
    r_outer: float,
    n_sectors: int,
    tri_list: List[np.ndarray],
    mat_ids: List[int],
    mat_emissive: int,
    mat_back: int,
    source_tri_ids: List[int],
) -> None:
    """Flat annular ring light flush with the front of the lens barrel.

    Scene-facing side (-X normal, toward subject) is emissive and registered as a
    light source.  Camera-body side (+X normal) is black matte — no light exits
    toward the sensor.  Camera looks in -X so the scene is at smaller X values.
    """
    n = max(6, int(n_sectors))
    for i in range(n):
        a0 = 2.0 * math.pi * i       / n
        a1 = 2.0 * math.pi * (i + 1) / n
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)
        # Four corners of the quad (in YZ plane at x_pos)
        p_i0 = np.array([x_pos, r_inner * ca0, r_inner * sa0], dtype=np.float64)
        p_i1 = np.array([x_pos, r_inner * ca1, r_inner * sa1], dtype=np.float64)
        p_o0 = np.array([x_pos, r_outer * ca0, r_outer * sa0], dtype=np.float64)
        p_o1 = np.array([x_pos, r_outer * ca1, r_outer * sa1], dtype=np.float64)
        # Scene-facing side: -X normal (toward subject, camera looks in -X).
        # Reversed winding so normal points in -X direction.
        source_tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, p_i1, p_o0, p_i0, mat_emissive)
        source_tri_ids.append(len(tri_list))
        _append_tri(tri_list, mat_ids, p_i1, p_o1, p_o0, mat_emissive)
        # Camera-body side: +X normal (black matte, no light exits toward sensor)
        _append_tri(tri_list, mat_ids, p_i0, p_o0, p_i1, mat_back)
        _append_tri(tri_list, mat_ids, p_o0, p_o1, p_i1, mat_back)


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
            gl_opacity=0.15,
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
            gl_opacity=0.35,
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
    idx_ring_light = db.register(
        "ring_light_emitter",
        Material(
            name="ring_light_emitter",
            domain="em_optical",
            albedo=[1.0, 0.98, 0.94],
            roughness=0.0,
            metallic=0.0,
            emission_rgb=[1.0, 0.98, 0.94],
            ior=1.0,
            transmission=0.0,
            radiance=RadianceProfile(
                luminance=8000.0,
                cct_k=5600.0,
                cri=98.0,
                solid_angle_sr=math.pi,
                distribution="lambertian",
            ),
            spectral_bands=_make_sidecar_spectral_bands(
                sidecar,
                reflectance=0.0,
                transmittance=0.0,
                diffuse_frac=0.0,
                emission_scale=float(getattr(scene, "ring_light_emission", 8.0)),
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
    diffuser_wave_specs: List[DiffuserWaveTubeSpec] = []

    if bool(getattr(scene, "include_legacy_stage", False)):
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
                wave_spec_out=diffuser_wave_specs,
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
    else:
        _import_subject_scene(scene, db, tris, mats, source_tri_ids, object_tri_ids)
    _iris = getattr(scene, "iris_aperture", None)
    _iris_active = _iris is not None and getattr(_iris, "enabled", False)
    if _iris_active:
        _build_iris_baffle(_iris, scene.tube_radius, tris, mats, idx_aperture_black, aperture_stop_tri_ids)

    if str(getattr(scene, "aperture_model", "geometry")).lower() == "wave3d":
        print(
            "[aperture-model] wave3d requested; geometric aperture mesh remains the boundary, "
            "wave aperture propagation is pending",
            flush=True,
        )
    if lenses:
        first_lens = lenses[0]
        last_lens = lenses[-1]
        # Physical camera body entrance = back face of the last lens group.
        # This is always driven by the actual lens geometry, not a heuristic or
        # the optical exit pupil (which may be a virtual pupil before G1).
        # exit_pupil_x  : where the light-tight barrel starts (G4 back face).
        # exit_pupil_radius : clear opening of the barrel at that face (G4 aperture).
        # Both are set unconditionally so they are correct before mesh triangles
        # are built below — set_optics() also writes them but runs after the mesh.
        scene.exit_pupil_x = round(float(last_lens.x_back) + 0.005, 6)
        scene.exit_pupil_radius = round(float(last_lens.aperture_radius), 6)
        bore_r = float(scene.tube_radius)
        lens_housing_r = float(last_lens.aperture_radius + 0.008)
        if bool(getattr(scene, "include_legacy_stage", False)):
            # Legacy pre-lens bore with CSG openings for authored light tubes.
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
        else:
            # Camera-owned lens hood: a conic frustum from the front mouth into
            # the first lens seat, with no imported stage walls or light tubes.
            hood_depth = float(min(0.030, max(0.010, 0.35 * first_lens.aperture_radius)))
            hood_x0 = float(first_lens.x_front - hood_depth)
            hood_x1 = float(first_lens.x_front - 0.002)
            hood_r1 = float(max(lens_housing_r, first_lens.aperture_radius + 0.004))
            _hood_depth = max(float(first_lens.x_front) - float(hood_x0), 1.0e-5)
            _tan_half_fov = float(scene.image_plate.radius) / max(float(scene.image_plate.x) - float(first_lens.x_front), 1.0e-5)
            _hood_r_fov = float(first_lens.aperture_radius) + _hood_depth * _tan_half_fov
            hood_clear_r0 = float(max(scene.lens_hood_front_radius, hood_r1 + 0.010, _hood_r_fov))
            hood_wall = float(max(0.004, 0.035 * hood_clear_r0))
            hood_r0 = float(hood_clear_r0 + hood_wall)
            _build_sensor_aperture_frustum(
                hood_x0,
                hood_r0,
                hood_x1,
                hood_r1,
                96,
                tris,
                mats,
                idx_aperture_black,
                tri_ids=black_wall_tri_ids,
            )
            _build_baffle_annulus(
                hood_x0,
                hood_clear_r0,
                hood_r0,
                96,
                tris,
                mats,
                idx_aperture_black,
                tri_ids=black_wall_tri_ids,
            )
            # ── Ring light: emissive annulus flush with the front face of the hood ──
            if bool(getattr(scene, "ring_light_enabled", True)):
                _rl_r_inner = hood_r0
                _rl_r_outer = _rl_r_inner + float(getattr(scene, "ring_light_width_m", 0.018))
                _rl_x = hood_x0
                _rl_n = int(getattr(scene, "ring_light_n_sectors", 72))
                _rl_before = len(tris)
                _build_ring_light(
                    x_pos=_rl_x,
                    r_inner=_rl_r_inner,
                    r_outer=_rl_r_outer,
                    n_sectors=_rl_n,
                    tri_list=tris,
                    mat_ids=mats,
                    mat_emissive=idx_ring_light,
                    mat_back=idx_black,
                    source_tri_ids=source_tri_ids,
                )
                print(
                    "[ring-light]",
                    f"x={_rl_x:.4f}",
                    f"r_inner={_rl_r_inner*1e3:.1f}mm",
                    f"r_outer={_rl_r_outer*1e3:.1f}mm",
                    f"sectors={_rl_n}",
                    f"emission_tris={len(tris)-_rl_before}",
                    flush=True,
                )
        # Lens housing sleeve: keeps the lens mechanically inset in a bore rather
        # than visually floating in open space.
        # Tighten the housing bore to last lens aperture to prevent rear light escape.
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

        if bool(getattr(scene, "include_legacy_stage", False)):
            # Legacy post-lens bellows.  The default camera-only path uses the
            # sensor enclosure's aperture-to-sensor frustum as the single rear cone.
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
    # Use actual lens housing radius (from group apertures), not the scene stage
    # tube radius — the camera barrel belongs to the lens assembly, not the stage.
    _tube_r    = float(lens_housing_r)
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

    # Debug leak-probe emitter removed: keep the plumbing arrays empty so the
    # optical scene is lit only by authored source geometry.
    red_probe_tri_ids: list = []

    tri_arr = np.ascontiguousarray(np.asarray(tris, dtype=np.float64))
    _auto_fit_scene_view_to_mesh(scene, tri_arr)
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
        diffuser_wave_specs,
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


BDPT_ENDPOINT_DTYPE = np.dtype([
    ("id", "u4"),          # forward: monotonic subpath id; backward: coarse legacy pixel id
    ("band", "u4"),
    ("pos", "f4", (3,)),
    ("amp", "c8"),
    ("film_uv", "f4", (2,)),
    ("launch_pdf", "f4"),  # fwd: area PDF [m⁻²]; bwd: exit solid-angle PDF [sr⁻¹]
    ("stream", "u1"),      # 0 = forward/light, 1 = backward/sensor
], align=False)
# NON-STANDARD BDPT NOTE:
# These endpoint rows are not Veach-style path vertices with full throughput,
# PDFs, measure conversions, and MIS data.  They are a compact cache used by
# this program's experimental connector: a sensor/light side label, one band,
# one endpoint position, a complex amplitude, and a sensor pixel/subpath id.
# That makes the cache useful for replay/debug and shadow-connection studies,
# but it is not sufficient by itself for an unbiased general BDPT estimator.

BDPT_FILM_TAG_FLAG = np.uint64(1) << np.uint64(63)
BDPT_FILM_TAG_UV_BITS = 30
BDPT_FILM_TAG_CHANNEL_SHIFT = np.uint64(60)
BDPT_FILM_TAG_UV_SHIFT = np.uint64(BDPT_FILM_TAG_UV_BITS)
BDPT_FILM_TAG_MASK = np.uint64((1 << BDPT_FILM_TAG_UV_BITS) - 1)


def _pack_bdpt_film_tag(channel_idx: np.ndarray, film_uv: np.ndarray) -> np.ndarray:
    uv = np.clip(np.asarray(film_uv, dtype=np.float64), 0.0, 1.0)
    q = float((1 << BDPT_FILM_TAG_UV_BITS) - 1)
    u = np.rint(uv[:, 0] * q).astype(np.uint64) & BDPT_FILM_TAG_MASK
    v = np.rint(uv[:, 1] * q).astype(np.uint64) & BDPT_FILM_TAG_MASK
    ch = (np.asarray(channel_idx, dtype=np.uint64) & np.uint64(0x3)) << BDPT_FILM_TAG_CHANNEL_SHIFT
    return BDPT_FILM_TAG_FLAG | ch | (u << BDPT_FILM_TAG_UV_SHIFT) | v


def _unpack_bdpt_film_tag(tags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(tags, dtype=np.uint64)
    valid = (t & BDPT_FILM_TAG_FLAG) != np.uint64(0)
    q = float((1 << BDPT_FILM_TAG_UV_BITS) - 1)
    uv = np.zeros((t.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (((t >> BDPT_FILM_TAG_UV_SHIFT) & BDPT_FILM_TAG_MASK).astype(np.float64) / q).astype(np.float32)
    uv[:, 1] = ((t & BDPT_FILM_TAG_MASK).astype(np.float64) / q).astype(np.float32)
    return valid, uv


def _smooth_gate(edge_value: np.ndarray, softness: float) -> np.ndarray:
    """Return 0..1 transmission for edge_value >= 0 with optional smooth edge."""
    e = np.asarray(edge_value, dtype=np.float64)
    s = float(max(0.0, softness))
    if s <= 1.0e-9:
        return (e >= 0.0).astype(np.float64)
    x = np.clip(0.5 + 0.5 * e / s, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def film_shutter_transmission(plate: ImagePlateConfig, film_uv: np.ndarray) -> np.ndarray:
    """Evaluate a simple film-domain shutter exposure mask.

    This is deliberately a film/sensor UV exposure gate, not lens physics.  It
    can model open/closed exposure, radial iris-like film masks, and sliding
    window shutters before we add time-resolved shutter mechanics.
    """
    uv = np.asarray(film_uv, dtype=np.float64)
    if uv.size == 0:
        return np.zeros((0,), dtype=np.float64)
    mode = str(getattr(plate, "shutter_mode", "open") or "open").lower()
    open_f = float(np.clip(getattr(plate, "shutter_open", 1.0), 0.0, 1.0))
    soft = float(max(0.0, getattr(plate, "shutter_softness", 0.0)))
    if mode == "open":
        return np.ones((uv.shape[0],), dtype=np.float64)
    if mode in ("closed", "cap", "lens_cap"):
        return np.zeros((uv.shape[0],), dtype=np.float64)
    if open_f <= 0.0:
        return np.zeros((uv.shape[0],), dtype=np.float64)
    if open_f >= 1.0 and mode in ("iris", "sliding_x", "sliding_y"):
        return np.ones((uv.shape[0],), dtype=np.float64)

    cu = float(np.clip(getattr(plate, "shutter_center_u", 0.5), 0.0, 1.0))
    cv = float(np.clip(getattr(plate, "shutter_center_v", 0.5), 0.0, 1.0))
    if mode == "iris":
        # Radius is expressed in normalized film coordinates where the full
        # sensor disc has radius 0.5.
        r = 0.5 * math.sqrt(open_f)
        d = np.sqrt((uv[:, 0] - cu) ** 2 + (uv[:, 1] - cv) ** 2)
        return _smooth_gate(r - d, soft)
    if mode == "sliding_x":
        half_w = 0.5 * open_f
        d = np.abs(uv[:, 0] - cu)
        return _smooth_gate(half_w - d, soft)
    if mode == "sliding_y":
        half_h = 0.5 * open_f
        d = np.abs(uv[:, 1] - cv)
        return _smooth_gate(half_h - d, soft)
    return np.ones((uv.shape[0],), dtype=np.float64)


BDPT_MIS_EPS = 1.0e-30


def assemble_bdpt_connection_weight(
    fa: np.ndarray,
    ba: np.ndarray,
    dist: np.ndarray,
    fwd_pdf: np.ndarray,
    bwd_pdf: np.ndarray,
    pair_scale: float,
) -> np.ndarray:
    """MIS-weighted contribution for connected forward/backward endpoint pairs.

    ``fwd_pdf`` is an emitter endpoint area PDF. ``bwd_pdf`` is the camera
    exit solid-angle PDF computed from the aperture→exit-direction Jacobian.
    For a connection to a scene endpoint, the camera PDF is converted to
    endpoint-area measure with dω ≈ dA / r². Endpoint surface cosine terms are
    still not available in BDPT_ENDPOINT_DTYPE.
    """
    d2  = np.maximum(dist * dist, 1.0e-12)
    p_b = np.maximum(bwd_pdf.astype(np.float64) / d2, BDPT_MIS_EPS)
    p_f = np.maximum(fwd_pdf.astype(np.float64), BDPT_MIS_EPS)
    # Balance heuristic: w_cam = p_bwd / (p_bwd + p_fwd)
    # Combined estimator: f * w_cam / p_bwd = f / (p_bwd + p_fwd)
    mis_denom = p_b + p_f
    raw = np.abs(fa * ba) / d2
    return raw * (pair_scale / mis_denom)


def mis_balance_weight(this_pdf: float, all_pdfs: Sequence[float]) -> float:
    """Balance-heuristic MIS weight for one sampling strategy."""
    denom = float(np.sum(np.maximum(np.asarray(all_pdfs, dtype=np.float64), 0.0)))
    return float(max(float(this_pdf), 0.0) / max(denom, BDPT_MIS_EPS))


def mis_power_weight(this_pdf: float, all_pdfs: Sequence[float], beta: float = 2.0) -> float:
    """Power-heuristic MIS weight.  This is the default hook for future BDPT strategies."""
    p = np.maximum(np.asarray(all_pdfs, dtype=np.float64), 0.0)
    w = np.power(p, float(beta))
    this = float(max(float(this_pdf), 0.0) ** float(beta))
    return float(this / max(float(np.sum(w)), BDPT_MIS_EPS))


BDPT_SEGMENT_DTYPE = np.dtype([
    ("subpath_id", "u4"),
    ("band", "u4"),
    ("p0", "f4", (3,)),
    ("p1", "f4", (3,)),
    ("dir", "f4", (3,)),
    ("amp", "c8"),
    ("film_uv", "f4", (2,)),
    ("path_len", "f4"),
    ("pdf_fwd", "f4"),
    ("pdf_rev", "f4"),
    ("pdf_area", "f4"),          # joint area-domain PDF (film_area × aperture_area for camera rays)
    ("pdf_solid_angle", "f4"),   # solid-angle PDF (0 when Jacobian is missing; see flags)
    ("strategy_id", "u2"),       # CAMERA_STRATEGY_SENSOR_APERTURE or 0
    ("sample_domain", "u1"),     # BDPT_SAMPLE_DOMAIN_*
    ("stream", "u1"),       # 0 = light/forward, 1 = camera/backward
    ("kind", "u1"),         # RayRecordKind from C++
    ("bounce", "i4"),
    ("hit_group_id", "i4"),
    ("hit_tri", "i4"),
    ("flags", "u4"),
], align=False)


class BdptSegmentStore:
    """Chunked subsegment/vertex stream for future BDPT/MIS estimators.

    Unlike ``BdptEndpointStore``, this keeps segment start/end geometry,
    directions, path length, bounce, surface identifiers, film coordinates, and
    PDF placeholders.  The current endpoint connector does not consume it yet;
    it is the modular handoff point for a real path-strategy/MIS estimator.
    """

    def __init__(self, max_rows: Optional[int] = 2_000_000) -> None:
        self.max_rows = None if max_rows is None else int(max_rows)
        self._chunks: list[np.ndarray] = []
        self._count = 0

    def append(self, rows: np.ndarray) -> None:
        if rows is None or rows.size == 0:
            return
        r = np.asarray(rows, dtype=BDPT_SEGMENT_DTYPE)
        if r.ndim != 1:
            return
        self._chunks.append(r)
        self._count += int(r.shape[0])
        if self.max_rows is not None:
            while self._count > self.max_rows and self._chunks:
                oldest = self._chunks[0]
                drop = min(int(oldest.shape[0]), self._count - self.max_rows)
                if drop >= int(oldest.shape[0]):
                    self._chunks.pop(0)
                    self._count -= int(oldest.shape[0])
                else:
                    self._chunks[0] = oldest[drop:]
                    self._count -= int(drop)

    def snapshot(self) -> Optional[np.ndarray]:
        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return np.ascontiguousarray(self._chunks[0], dtype=BDPT_SEGMENT_DTYPE)
        return np.ascontiguousarray(np.concatenate(self._chunks, axis=0), dtype=BDPT_SEGMENT_DTYPE)

    def count(self) -> int:
        return int(self._count)

    def clear(self) -> None:
        self._chunks.clear()
        self._count = 0


OPTICAL_TRANSFER_DTYPE = np.dtype([
    ("batch_id", "u4"),
    ("ray_id", "u4"),
    ("event_index", "u2"),
    ("element_idx", "i4"),
    ("kind", "u1"),          # 0 conic, 1 flat, 2 aperture/stop, 255 other
    ("reason", "u1"),        # TerminationReason.value
    ("p0", "f4", (3,)),
    ("p1", "f4", (3,)),
    ("dir_in", "f4", (3,)),
    ("dir_out", "f4", (3,)),
    ("normal", "f4", (3,)),
    ("geom_len", "f4"),
    ("opl", "f4"),
    ("n_before", "f4"),
    ("n_after", "f4"),
    ("aperture_r", "f4"),
    ("film_uv", "f4", (2,)),
    ("fresnel_r",   "f4"),   # unpolarized Fresnel reflectance (0 for stops)
    ("fresnel_t",   "f4"),   # transmittance = 1 - fresnel_r (0 for failed/TIR)
    ("eta_ratio",   "f4"),   # n_before / n_after
    ("cos_i",       "f4"),   # cosine of incidence angle
    ("cos_t",       "f4"),   # cosine of refraction angle (0 for TIR/stops)
    ("throughput",  "f4"),   # fresnel_t if passed, 0 if blocked/TIR
    ("strategy_id", "u2"),   # CAMERA_STRATEGY_SENSOR_APERTURE or 0
    ("sample_domain", "u1"), # BDPT_SAMPLE_DOMAIN_*
    ("stream", "u1"),        # 0 forward, 1 backward/camera
    ("flags", "u4"),
], align=False)


CAMERA_SAMPLE_DTYPE = np.dtype([
    ("batch_id", "u4"),
    ("sample_id", "u4"),
    ("film_uv", "f4", (2,)),
    ("film_pos", "f4", (3,)),
    ("aperture_pos", "f4", (3,)),
    ("dir", "f4", (3,)),
    ("sensor_channel", "u1"),
    ("n_bands", "u2"),
    ("shutter_weight", "f4"),
    ("sensor_weight", "f4", (MAX_SPECTRAL_BANDS,)),
    ("film_area_pdf", "f4"),
    ("aperture_area_pdf", "f4"),
    ("channel_pdf", "f4"),
    ("strategy_pdf", "f4"),
    ("aperture_to_solid_angle_jac", "f4"),  # |dω_exit / dA_aperture|
    ("solid_angle_pdf", "f4"),              # aperture_area_pdf / aperture jac
    ("phase_space_jac", "f4"),              # |d(exit_y,z,dir_y,z) / d(film_y,z,ap_y,z)|
    ("phase_space_pdf", "f4"),              # film_aperture_area_pdf / phase_space_jac
    ("strategy_id", "u2"),
    ("flags", "u4"),
], align=False)


_OPTICAL_TRANSFER_KIND_NAMES = {
    0: "conic",
    1: "flat",
    2: "stop",
    255: "other",
}


CAMERA_STRATEGY_SENSOR_APERTURE = 1
CAMERA_SAMPLE_FLAG_FILM_PDF_APPROX = np.uint32(1 << 0)
CAMERA_SAMPLE_FLAG_JACOBIAN_VALID = np.uint32(1 << 1)

# Segment flags
BDPT_SEGMENT_FLAG_JACOBIAN_MISSING = np.uint32(1 << 1)  # camera-to-scene Jacobian not available

# Sample domain codes stored in segment / optical-transfer records
BDPT_SAMPLE_DOMAIN_UNKNOWN       = np.uint8(0)
BDPT_SAMPLE_DOMAIN_HEMISPHERE    = np.uint8(1)   # cosine-hemisphere (forward emitter)
BDPT_SAMPLE_DOMAIN_FILM_APERTURE = np.uint8(2)   # film UV × aperture disc (backward camera)


def _fill_optical_transfer_fresnel(rows: np.ndarray) -> None:
    """Compute and write Fresnel/throughput fields into rows in-place."""
    n1  = rows["n_before"].astype(np.float64)
    n2  = rows["n_after"].astype(np.float64)
    d   = rows["dir_in"].astype(np.float64)    # (N, 3)
    nm  = rows["normal"].astype(np.float64)    # (N, 3)

    cos_i_raw = -np.einsum("ij,ij->i", d, nm)
    # Match _snell(): when the ray arrives from the opposite side, the
    # interface normal is flipped before evaluating Fresnel terms.
    cos_i = np.clip(np.abs(cos_i_raw), 0.0, 1.0)
    eta   = n1 / np.maximum(n2, 1.0e-9)
    sin2t = eta * eta * (1.0 - cos_i * cos_i)
    tir   = sin2t >= 1.0
    cos_t = np.where(tir, 0.0, np.sqrt(np.maximum(0.0, 1.0 - sin2t)))

    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n1 * cos_t + n2 * cos_i
    rs = np.where(denom_s > 1.0e-12, (n1 * cos_i - n2 * cos_t) / denom_s, 1.0)
    rp = np.where(denom_p > 1.0e-12, (n1 * cos_t - n2 * cos_i) / denom_p, 1.0)
    r  = np.clip(0.5 * (rs * rs + rp * rp), 0.0, 1.0)
    r  = np.where(tir, 1.0, r)

    # Aperture/stop surfaces: no Fresnel, geometry-only
    is_stop   = rows["kind"].astype(np.int32) == 2
    r         = np.where(is_stop, 0.0, r)
    t         = 1.0 - r
    is_passed = rows["reason"].astype(np.int32) == int(TerminationReason.PASSED.value)

    rows["fresnel_r"]  = r.astype(np.float32)
    rows["fresnel_t"]  = t.astype(np.float32)
    rows["eta_ratio"]  = eta.astype(np.float32)
    rows["cos_i"]      = cos_i.astype(np.float32)
    rows["cos_t"]      = cos_t.astype(np.float32)
    rows["throughput"] = np.where(is_passed, t, 0.0).astype(np.float32)


def summarize_optical_transfer_rows(rows: np.ndarray) -> Dict[str, object]:
    """Compact diagnostics for one batch of parametric optical transfer events."""
    if rows is None or rows.size == 0:
        return {
            "events": 0,
            "rays": 0,
            "failed_rays": 0,
            "passed_events": 0,
            "failed_events": 0,
        }
    r = np.asarray(rows, dtype=OPTICAL_TRANSFER_DTYPE)
    reason_names = {int(v.value): str(v.name) for v in TerminationReason}
    passed_code = int(TerminationReason.PASSED.value)
    reason = r["reason"].astype(np.int32, copy=False)
    ray_ids = r["ray_id"].astype(np.uint32, copy=False)
    fail = reason != passed_code
    unique_rays = np.unique(ray_ids)
    failed_rays = np.unique(ray_ids[fail]) if np.any(fail) else np.zeros(0, dtype=np.uint32)

    reason_counts: Dict[str, int] = {}
    for code in np.unique(reason):
        c = int(code)
        reason_counts[reason_names.get(c, f"reason_{c}")] = int(np.count_nonzero(reason == c))

    top_fail_element = -1
    top_fail_reason = ""
    top_fail_kind = ""
    top_fail_count = 0
    if np.any(fail):
        failed = r[fail]
        keys = np.stack([
            failed["element_idx"].astype(np.int32, copy=False),
            failed["reason"].astype(np.int32, copy=False),
            failed["kind"].astype(np.int32, copy=False),
        ], axis=1)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        imax = int(np.argmax(counts))
        top = uniq[imax]
        top_fail_element = int(top[0])
        top_fail_reason = reason_names.get(int(top[1]), f"reason_{int(top[1])}")
        top_fail_kind = _OPTICAL_TRANSFER_KIND_NAMES.get(int(top[2]), f"kind_{int(top[2])}")
        top_fail_count = int(counts[imax])

    opl = r["opl"].astype(np.float64, copy=False)
    geom = r["geom_len"].astype(np.float64, copy=False)
    return {
        "events": int(r.shape[0]),
        "rays": int(unique_rays.shape[0]),
        "failed_rays": int(failed_rays.shape[0]),
        "passed_events": int(np.count_nonzero(~fail)),
        "failed_events": int(np.count_nonzero(fail)),
        "reason_counts": reason_counts,
        "top_fail_element": top_fail_element,
        "top_fail_reason": top_fail_reason,
        "top_fail_kind": top_fail_kind,
        "top_fail_count": top_fail_count,
        "opl_min": float(np.min(opl)) if opl.size else 0.0,
        "opl_mean": float(np.mean(opl)) if opl.size else 0.0,
        "opl_max": float(np.max(opl)) if opl.size else 0.0,
        "geom_mean": float(np.mean(geom)) if geom.size else 0.0,
    }


def summarize_camera_sample_rows(rows: np.ndarray) -> Dict[str, object]:
    """Compact diagnostics for one camera launch sample batch."""
    if rows is None or rows.size == 0:
        return {"samples": 0}
    r = np.asarray(rows, dtype=CAMERA_SAMPLE_DTYPE)
    strat = r["strategy_pdf"].astype(np.float64, copy=False)
    solid = r["solid_angle_pdf"].astype(np.float64, copy=False)
    phase = r["phase_space_pdf"].astype(np.float64, copy=False)
    jac = r["aperture_to_solid_angle_jac"].astype(np.float64, copy=False)
    ps_jac = r["phase_space_jac"].astype(np.float64, copy=False)
    shutter = r["shutter_weight"].astype(np.float64, copy=False)
    ch = r["sensor_channel"].astype(np.int32, copy=False)
    channel_counts: Dict[str, int] = {}
    for c in np.unique(ch):
        channel_counts[str(int(c))] = int(np.count_nonzero(ch == c))
    return {
        "samples": int(r.shape[0]),
        "strategy_pdf_min": float(np.min(strat)) if strat.size else 0.0,
        "strategy_pdf_mean": float(np.mean(strat)) if strat.size else 0.0,
        "strategy_pdf_max": float(np.max(strat)) if strat.size else 0.0,
        "solid_angle_pdf_min": float(np.min(solid[solid > 0.0])) if np.any(solid > 0.0) else 0.0,
        "solid_angle_pdf_mean": float(np.mean(solid[solid > 0.0])) if np.any(solid > 0.0) else 0.0,
        "solid_angle_pdf_max": float(np.max(solid)) if solid.size else 0.0,
        "phase_space_pdf_min": float(np.min(phase[phase > 0.0])) if np.any(phase > 0.0) else 0.0,
        "phase_space_pdf_mean": float(np.mean(phase[phase > 0.0])) if np.any(phase > 0.0) else 0.0,
        "phase_space_pdf_max": float(np.max(phase)) if phase.size else 0.0,
        "jacobian_valid": int(np.count_nonzero((jac > 0.0) & (ps_jac > 0.0))),
        "shutter_min": float(np.min(shutter)) if shutter.size else 0.0,
        "shutter_mean": float(np.mean(shutter)) if shutter.size else 0.0,
        "shutter_max": float(np.max(shutter)) if shutter.size else 0.0,
        "channel_counts": channel_counts,
    }


class CameraSampleStore:
    """Camera-side birth records for backward paths and future PDF/MIS work."""

    def __init__(self, max_rows: Optional[int] = 2_000_000) -> None:
        self.max_rows = None if max_rows is None else int(max_rows)
        self._chunks: list[np.ndarray] = []
        self._count = 0

    def append(self, rows: np.ndarray) -> None:
        if rows is None or rows.size == 0:
            return
        r = np.asarray(rows, dtype=CAMERA_SAMPLE_DTYPE)
        if r.ndim != 1:
            return
        self._chunks.append(r)
        self._count += int(r.shape[0])
        if self.max_rows is not None:
            while self._count > self.max_rows and self._chunks:
                oldest = self._chunks[0]
                drop = min(int(oldest.shape[0]), self._count - self.max_rows)
                if drop >= int(oldest.shape[0]):
                    self._chunks.pop(0)
                    self._count -= int(oldest.shape[0])
                else:
                    self._chunks[0] = oldest[drop:]
                    self._count -= int(drop)

    def snapshot(self) -> Optional[np.ndarray]:
        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return np.ascontiguousarray(self._chunks[0], dtype=CAMERA_SAMPLE_DTYPE)
        return np.ascontiguousarray(np.concatenate(self._chunks, axis=0), dtype=CAMERA_SAMPLE_DTYPE)

    def count(self) -> int:
        return int(self._count)

    def clear(self) -> None:
        self._chunks.clear()
        self._count = 0


class OpticalTransferStore:
    """Parametric lens event records emitted before/after GPU scene tracing."""

    def __init__(self, max_rows: Optional[int] = 2_000_000) -> None:
        self.max_rows = None if max_rows is None else int(max_rows)
        self._chunks: list[np.ndarray] = []
        self._count = 0

    def append(self, rows: np.ndarray) -> None:
        if rows is None or rows.size == 0:
            return
        r = np.asarray(rows, dtype=OPTICAL_TRANSFER_DTYPE)
        if r.ndim != 1:
            return
        self._chunks.append(r)
        self._count += int(r.shape[0])
        if self.max_rows is not None:
            while self._count > self.max_rows and self._chunks:
                oldest = self._chunks[0]
                drop = min(int(oldest.shape[0]), self._count - self.max_rows)
                if drop >= int(oldest.shape[0]):
                    self._chunks.pop(0)
                    self._count -= int(oldest.shape[0])
                else:
                    self._chunks[0] = oldest[drop:]
                    self._count -= int(drop)

    def snapshot(self) -> Optional[np.ndarray]:
        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return np.ascontiguousarray(self._chunks[0], dtype=OPTICAL_TRANSFER_DTYPE)
        return np.ascontiguousarray(np.concatenate(self._chunks, axis=0), dtype=OPTICAL_TRANSFER_DTYPE)

    def count(self) -> int:
        return int(self._count)

    def clear(self) -> None:
        self._chunks.clear()
        self._count = 0


class BdptEndpointStore:
    """Persistent accumulator for BDPT half-path endpoints.

    Stores the smallest package needed by the Python BDPT connector:
    id/pixel, band, endpoint position, complex amplitude, and stream side.

    Both forward and backward endpoints are retained for the whole run.  The
    image integrator is progressive, so dropping old endpoints would erase
    valid transport history.
    """

    def __init__(self) -> None:
        self._fwd: list[np.ndarray] = []   # each entry is a BDPT_ENDPOINT_DTYPE chunk
        self._bwd: list[np.ndarray] = []
        self._fwd_count: int = 0
        self._bwd_count: int = 0

    # ------------------------------------------------------------------
    def append(self, new_rows: np.ndarray) -> None:
        """Add *new_rows* (BDPT_ENDPOINT_DTYPE, shape (M,)) to the store."""
        if new_rows is None or new_rows.size == 0:
            return
        rows = np.asarray(new_rows, dtype=BDPT_ENDPOINT_DTYPE)
        if rows.ndim != 1:
            return
        sid = rows["stream"]
        fwd = rows[sid == 0]
        bwd = rows[sid == 1]
        if fwd.shape[0] > 0:
            self._fwd.append(fwd)
            self._fwd_count += fwd.shape[0]
        if bwd.shape[0] > 0:
            self._bwd.append(bwd)
            self._bwd_count += bwd.shape[0]

    def snapshot(self) -> Optional[np.ndarray]:
        """Return a single contiguous array of all current endpoints."""
        parts = []
        if self._fwd:
            parts.append(np.concatenate(self._fwd, axis=0) if len(self._fwd) > 1 else self._fwd[0])
        if self._bwd:
            parts.append(np.concatenate(self._bwd, axis=0) if len(self._bwd) > 1 else self._bwd[0])
        if not parts:
            return None
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=BDPT_ENDPOINT_DTYPE)

    def fwd_count(self) -> int:
        return self._fwd_count

    def bwd_count(self) -> int:
        return self._bwd_count

    def clear(self) -> None:
        self._fwd.clear()
        self._bwd.clear()
        self._fwd_count = 0
        self._bwd_count = 0


def _bdpt_compact_to_legacy_records(records: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Expand compact endpoint records for older debug/export helpers."""
    if records is None or records.size == 0:
        return None
    rec = np.asarray(records, dtype=BDPT_ENDPOINT_DTYPE)
    out = np.zeros((rec.shape[0], 16), dtype=np.float32)
    out_u32 = out.view(np.uint32)
    out_i32 = out.view(np.int32)
    out_u32[:, 0] = rec["id"]
    out_u32[:, 1] = rec["band"]
    out_i32[:, 3] = np.where(rec["stream"] == 1, 0, -1).astype(np.int32, copy=False)
    out[:, 4:7] = rec["pos"]
    out[:, 11] = 1.0
    out[:, 12] = rec["amp"].real.astype(np.float32, copy=False)
    out[:, 13] = rec["amp"].imag.astype(np.float32, copy=False)
    out[:, 14] = 1.0
    out[:, 15] = rec["stream"].astype(np.float32, copy=False)
    bwd = rec["stream"] == 1
    if np.any(bwd):
        uv = np.clip(rec["film_uv"][bwd], 0.0, 1.0 - 1.0e-7)
        legacy_res = 65535.0
        row = np.floor(uv[:, 1] * legacy_res).astype(np.uint32)
        col = np.floor(uv[:, 0] * legacy_res).astype(np.uint32)
        out_u32[bwd, 0] = row * np.uint32(65535) + col
    return out


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
        self._mobile_lens_idx: int = 0
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
        self.bdpt_last_backward_transport: Dict[str, int] = {}
        self.bdpt_last_optical_transfer: Dict[str, object] = {}
        self.bdpt_last_camera_samples: Dict[str, object] = {}
        self._aperture_aim_extrema: dict = {}
        self.bdpt_last_stream_counts: Dict[str, int] = {
            "forward_light": 0, "backward_sensor": 0,
            "pixel_cone": 0, "field_deposit_blocked": 0,
        }
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
        self._last_bdpt_plate_rgb: Optional[np.ndarray] = None
        self._bdpt_plate_linear_accum: Optional[np.ndarray] = None
        self._bdpt_plate_weight_accum: Optional[np.ndarray] = None
        self._bdpt_plate_sample_count: int = 0
        self._bdpt_bwd_cursor: int = 0
        self._bdpt_endpoints: BdptEndpointStore = BdptEndpointStore()
        self._bdpt_segments: BdptSegmentStore = BdptSegmentStore(max_rows=2_000_000)
        self._optical_transfers: OpticalTransferStore = OpticalTransferStore(max_rows=2_000_000)
        self._camera_samples: CameraSampleStore = CameraSampleStore(max_rows=2_000_000)
        self._camera_sample_batch_id: int = 0
        self._last_backward_film_pdf: float = 0.0
        self._last_backward_aperture_pdf: float = 0.0
        self._last_backward_channel_pdf: float = 0.0
        self._last_backward_strategy_pdf: float = 0.0
        self._last_backward_cos_exit: float = 1.0   # mean |d_exit · x_hat| across passed rays
        self._emitter_launch_pdf: float = 0.0
        self._backward_launch_pdf_by_tag: Dict[int, float] = {}
        self._backward_area_pdf_by_tag: Dict[int, float] = {}
        self._backward_jacobian_by_tag: Dict[int, float] = {}
        self._async_bdpt_next_subpath: int = 0
        self._async_bdpt_next_segment_subpath: int = 0
        self._sweep_pixel_offset: int = 0   # deterministic scan position across disc pixels
        self._async_bdpt_forward_warmup_target: int = 0
        self._async_bdpt_forward_warmup_batch: int = 0
        self._async_forward_strike_count: int = 0
        self._async_forward_lens_hit_count: int = 0
        self._async_forward_lens_hit_after_bounce_count: int = 0
        self._async_forward_launched_count: int = 0
        self._async_backward_attempt_count: int = 0
        self._async_backward_parametric_absorbed_count: int = 0
        self._async_backward_launched_count: int = 0
        self._async_backward_strike_count: int = 0
        self._async_backward_skip_reason: str = ""
        self._bdpt_shadow_batch_cap: int = 1_000_000
        self._bdpt_shadow_max_fwd_per_bwd: int = 8
        self._bdpt_shadow_tag_base: int = 0xBD00000000000000
        self.bdpt_last_shadow_stats: Dict[str, float] = {}
        # Lens assembly descriptor — owns camera representation state (NONE | LUT | MLP),
        # registration, and rendering.  Replaces the former scattered _neural_assembly_*,
        # and _transfer_grid attributes.
        self._lens_assembly: Optional[LensAssemblySpec] = None
        # GPU/CPU compute mode: 'gpu', 'cpu', or 'mixed'.
        # Controls use_gpu_compute and gpu_all_stages in submit_rays.
        self.compute_mode: str = "gpu"
        # Persistent-pipeline drain loop
        self._drain_thread: Optional[threading.Thread] = None
        self._drain_stop  = threading.Event()
        self._segs_lock   = threading.Lock()
        # Bounded ring buffer of raw hit positions for the point overlay.
        # Each row: (x, y, z, amplitude, display_class, hit_group_id).
        # 2M rows × 6 floats × 4 bytes = 48 MB.  No spatial quantization.
        _VIS_CAP          = BDPT_PIPELINE_CAP
        self._VIS_CAP     = _VIS_CAP
        self._vis_buf     = np.zeros((_VIS_CAP, 6), dtype=np.float32)
        self._gid_crossing_counts: dict = {}  # gid → hit count in current ring window
        self._vis_ptr     = 0     # next write position (ring head)
        self._vis_full    = False  # True once ring has wrapped at least once
        self._forward_img_res = int(max(16, int(self.scene.image_plate.sensor_res)))
        self._forward_img_accum = np.zeros((self._forward_img_res, self._forward_img_res, 3), dtype=np.float32)
        self._forward_img_gain = 0.035
        self._reverse_img_accum = np.zeros_like(self._forward_img_accum)
        self._reverse_img_gain = 0.035
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
        # Cull rays with non-finite (inf/NaN) origins or directions before GPU
        # submission.  Default on; set False only for debugging.
        self.cull_infinite_rays: bool = True
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
            red_probe_ids, diffuser_wave_specs,
        ) = _build_scene_mesh(self.scene, self.sidecar)
        self._diffuser_wave_specs: List[DiffuserWaveTubeSpec] = list(diffuser_wave_specs)
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

        if self._lens_assembly is None:
            self._lens_assembly = LensAssemblySpec()
        self._lens_assembly.set_optics(
            _compound_lens_from_scene(self.scene),
            mode=LensAssemblySpec.MODE_PARAMETRIC,
        )
        # Assembly now owns the sensor plate and iris aperture.  All backward-ray
        # geometry derives from the assembly's actual part positions, not from
        # the surrogate solver's heuristic exit-pupil fields.
        self._lens_assembly.sync_from_scene(self.scene)

        # ── Optical exit pupil: where backward rays must be AIMED ─────────────
        # This can be a virtual pupil (x < G1) for telephoto layouts — physically
        # correct.  It drives aperture_centroid/radius (the ray-direction target)
        # but must NOT drive physical camera-body mesh positions.
        _ap_cen, _ap_r = self._lens_assembly.backward_ray_target()
        if _ap_cen is not None and _ap_r > 0.0:
            self.aperture_centroid = np.asarray(_ap_cen, dtype=np.float64)
            self.aperture_radius = float(_ap_r)
            print(
                "[assembly-optical-ep]",
                f"x={float(_ap_cen[0]):.4f}",
                f"r={_ap_r*1e3:.2f}mm",
                flush=True,
            )

        # ── Physical camera body: where the barrel geometry lives ─────────────
        # Always the back face of the last lens group (G4), never the optical EP.
        # scene.exit_pupil_x is the anchor for all camera-body mesh construction.
        _body_x = self._lens_assembly.camera_body_front_x()
        _body_r = self._lens_assembly.camera_body_entrance_radius()
        if _body_x is not None and _body_x > 0.0:
            self.scene.exit_pupil_x = float(_body_x)
            if _body_r > 0.0:
                self.scene.exit_pupil_radius = float(_body_r)
            print(
                "[assembly-physical-body]",
                f"body_x={float(_body_x):.4f}",
                f"body_r={_body_r*1e3:.2f}mm",
                f"sensor_x={self._lens_assembly.sensor_x()}",
                flush=True,
            )
        self._report_aperture_aim_extrema()
        _field_pair = self._lens_assembly.profile_field_pair(
            object_point=(float(self.scene.object_plane.x), 0.0, 0.0),
            image_point=(float(self.scene.image_plate.x), 0.0, 0.0),
            verify=False,
        )
        self._optical_field_pair = _field_pair
        _front_lim = _field_pair["front"].limiting_face
        _back_lim = _field_pair.get("back").limiting_face if _field_pair.get("back") is not None else None
        print(
            "[parametric-field]",
            f"faces={len(self._lens_assembly.optics.registered_faces())}",
            f"front_limit={_front_lim.face.element_idx if _front_lim is not None else -1}",
            f"front_cutoff={_field_pair['front'].cutoff_half_angle_rad:.4f}rad",
            f"back_limit={_back_lim.face.element_idx if _back_lim is not None else -1}",
            f"back_cutoff={_field_pair['back'].cutoff_half_angle_rad:.4f}rad",
            flush=True,
        )
        
        # Suppress walls, lens faces, and FIELD_EXEMPT sources from the surface
        # overlay so the tone-map white point is set by downstream surfaces.
        self._surface_suppress_tri_ids = np.ascontiguousarray(
            np.concatenate([
                tube_wall_ids,
                lens_front_ids,
                lens_back_ids,
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

        # Drive forward tracing from authored emissive source geometry.
        self.src_pos = np.ascontiguousarray(self.tri_centroids[self.emitter_tri_ids], dtype=np.float64)
        src_n = int(self.src_pos.shape[0])
        src_normals = np.ascontiguousarray(normals[self.emitter_tri_ids], dtype=np.float64)
        self.src_dir = np.ascontiguousarray(src_normals, dtype=np.float64)
        # Emitter area PDF: uniform area sampling over all emitter triangles.
        _ev = self.tri_vertices[self.emitter_tri_ids]   # (N, 3, 3)
        _e0 = _ev[:, 1, :] - _ev[:, 0, :]
        _e1 = _ev[:, 2, :] - _ev[:, 0, :]
        _tri_areas = 0.5 * np.linalg.norm(np.cross(_e0, _e1), axis=1)
        _total_emitter_area = float(np.sum(_tri_areas))
        self._emitter_launch_pdf: float = 1.0 / max(_total_emitter_area, 1.0e-30)
        # Neutral launch profile: no Python-side beaming. Keep transport driven
        # by emissive materials and scene geometry only.
        self.src_directivity = np.ones((src_n,), dtype=np.float64)
        self.tri_flux = np.zeros((self.n_tris, self.n_bands), dtype=np.float32)
        print(
            "[emitter-tris]",
            f"source={self.source_tri_ids.size}",
            f"debug_probe=removed",
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
        for i, (front_ids, back_ids, _rf, _rb) in enumerate(getattr(self, "lens_surface_groups", [])):
            bank.add(f"lens_{i:02d}_front", "lens_front", front_ids, self._uv_coords_for_tri_ids(front_ids, "yz"))
            bank.add(f"lens_{i:02d}_back", "lens_back", back_ids, self._uv_coords_for_tri_ids(back_ids, "yz"))
        self.uv_page_bank = bank
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
        """Register lens surfaces with the GPU tracer via LensAssemblySpec.

        In NONE/LUT mode: SDF_SPHERE parametric refinement on all lens surfaces.
        In MLP mode: NEURAL_ASSEMBLY entrance/exit + interior absorbers.
        Assembly is created here the first time (mode=NONE) if not already set.
        """
        lsg = getattr(self, "lens_surface_groups", [])
        if not lsg:
            return 0
        if self._lens_assembly is None:
            self._lens_assembly = LensAssemblySpec()
        self._lens_assembly.register(
            self.tracer,
            lsg,
            self.tri_vertices,
            self.tri_centroids,
            _scene_lenses(self.scene),
        )
        return len(lsg)

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
        if self._lens_assembly is not None:
            _bdpt_cen, _bdpt_r = self._lens_assembly.backward_ray_target()
            if _bdpt_cen is not None and _bdpt_r > 0.0:
                stop_plane_x = float(_bdpt_cen[0])
                stop_radius_m = float(_bdpt_r)

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
            # No explicit aperture mesh → hemisphere launch.  Nearly all rays will
            # be absorbed by the camera body frustum before reaching the scene.
            # Fix: place an IrisApertureConfig inside the lens assembly so the
            # pixel-cone can aim through the stop.  This is done automatically by
            # _scene_lenses() for four-group zoom surrogate designs.
            print(
                "[bdpt-no-aperture] aperture_stop_tri_ids=0"
                " -> hemisphere launch; backward rays will have ~0 emissive hits."
                " Ensure scene.iris_aperture is configured inside the lens stack.",
                flush=True,
            )
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
                    "n_aperture_samples": int(max(self.n_bands, n_aperture_samples)),
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
        # Re-register assembly groups if a payload or parametric model was previously loaded.
        self._do_register_neural_assembly_group()
        self._do_register_parametric_assembly()
        self._do_register_lut_assembly()

        self._bdpt_sensor_cfg = cfg
        print(
            "[bdpt-register]",
            f"source_gid={self._bdpt_source_gid}",
            f"debug_probe=removed",
            f"sensor_gid={self.bdpt_last_sensor_gid}",
            f"source_tris={int(self.source_tri_ids.size)}",
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
        self._register_wave_tubes()
        return int(self.bdpt_last_sensor_gid)

    def _register_wave_tubes(self) -> None:
        """Register a WaveTube surrogate emitter for every diffuser disc in the scene.

        Called at the end of _configure_sensor_film_pipeline() so groups survive
        clear_tri_groups().  Wave tubes are stored in self._wave_tubes.
        """
        from camera_designer.wave_tube import WaveTube, WaveTubeConfig
        specs = getattr(self, "_diffuser_wave_specs", [])
        wavelengths_m = (C_LIGHT / np.maximum(
            np.asarray(self.freq_hz, dtype=np.float64), 1.0
        )).astype(np.float64)
        self._wave_tubes: List[WaveTube] = []
        for spec in specs:
            if not int(spec.entry_tri_ids.size) or not int(spec.exit_tri_ids.size):
                continue
            cfg = WaveTubeConfig(
                axis            = spec.axis,
                entry_pos       = spec.entry_pos,
                exit_pos        = spec.exit_pos,
                tube_radius_m   = spec.tube_radius_m,
                n_medium        = spec.ior_real,
                n_imag          = spec.ior_imag,
                wavelengths_m   = wavelengths_m,
                nx              = 64,
                ny              = 64,
                dx_m            = 0.0,
                n_bpm_steps     = 0,
                pre_roll_frames = 8,
            )
            # Exit face centroids for BPM→tri_illum_accum spatial mapping.
            exit_cents = np.mean(
                self.tri_vertices[spec.exit_tri_ids], axis=1
            ).astype(np.float64)
            try:
                wt = WaveTube.register(
                    self.tracer,
                    spec.entry_tri_ids,
                    spec.exit_tri_ids,
                    cfg,
                    exit_tri_centroids=exit_cents,
                )
                self._wave_tubes.append(wt)
            except Exception as exc:
                print(f"[wave-tube] registration failed: {exc}", flush=True)
        # Ensure tri_illum_accum is sized for the full scene so write_tri_illum
        # has a valid buffer to target.
        if self._wave_tubes:
            try:
                self.tracer.init_illum_accum()
            except Exception:
                pass
        if self._wave_tubes:
            print(
                "[wave-tube]",
                f"registered={len(self._wave_tubes)}",
                f"pre_roll_frames={self._wave_tubes[0].config.pre_roll_frames}",
                "exit_role=pending_write_api",
                flush=True,
            )

    def solve_wave_tubes(self) -> int:
        """Advance all wave-tube BPM solvers by one frame.

        During pre-roll the accumulator warms up; on the pre-roll completion
        frame it is cleared so the live integral starts from zero.  After
        pre-roll each call propagates the captured entry field through the ADI
        BPM and stores the exit irradiance in wt._exit_field for downstream
        use (e.g., writing into tri_illum_accum when the API is available).

        Returns the number of tubes that are live (past pre-roll).
        """
        tubes = getattr(self, "_wave_tubes", [])
        if not tubes:
            return 0
        n_live = 0
        for wt in tubes:
            try:
                ef = wt.solve(self.tracer)
            except Exception as exc:
                print(f"[wave-tube] solve error: {exc}", flush=True)
                continue
            if ef is not None and wt._frames_collected > wt.config.pre_roll_frames:
                n_live += 1
        return n_live

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
        blocking: bool = False,
        bake_origins: Optional[np.ndarray] = None,
        bake_directions: Optional[np.ndarray] = None,
    ):
        """Forward trace via the persistent T1/T2/T3/T4 pipeline.

        Fans cosine-hemisphere rays from each source triangle and submits them
        to the machine immediately.  Returns the number of intents submitted.
        Output records arrive asynchronously via the background drain loop and
        are accumulated into tri_flux and _last_ray_segments.

        When ``blocking=True`` drains synchronously and returns the full records
        dict instead of the submitted count.  Pass ``bake_origins`` /
        ``bake_directions`` to bypass emitter sampling entirely (sequential tags,
        single child per ray — intended for baking training data).
        """
        # Advance wave-tube BPM solvers before submitting the next ray batch.
        # Uses the entry-accumulator data collected during the previous drain
        # cycle.  Bake calls bypass this (bake_origins implies a controlled
        # single-pass trace that should not disturb the wave-tube state).
        if bake_origins is None:
            self.solve_wave_tubes()

        _use_gpu = self.compute_mode in ("gpu", "mixed")
        _all_gpu = self.compute_mode == "gpu"

        if bake_origins is not None:
            n_submit      = len(bake_origins)
            origins       = np.ascontiguousarray(bake_origins,     dtype=np.float64)
            directions    = np.ascontiguousarray(bake_directions,   dtype=np.float64)
            amplitudes    = np.ones((n_submit, self.n_bands), dtype=np.complex128)
            src_ids       = np.zeros(n_submit, dtype=np.int32)
            tag_arr       = np.arange(n_submit, dtype=np.uint64)
            cflag_arr     = np.zeros(n_submit, dtype=np.uint8)
            total         = n_submit
            _max_children = 1
        else:
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

            _max_children = 2

        if self.cull_infinite_rays:
            origins, directions, amplitudes, src_ids, cflag_arr, tag_arr = _cull_finite_rays(
                origins, directions, amplitudes, src_ids, cflag_arr, tag_arr, label="fwd")

        self.tracer.submit_rays(
            origins=np.ascontiguousarray(origins),
            directions=np.ascontiguousarray(directions),
            amplitudes=np.ascontiguousarray(amplitudes),
            src_ids=np.ascontiguousarray(src_ids),
            tags=np.ascontiguousarray(tag_arr) if (tags is not None or bake_origins is not None) else None,
            color_flags=np.ascontiguousarray(cflag_arr),
            max_bounces=int(max_bounces),
            min_amplitude=float(self._min_amplitude),
            max_children=_max_children,
            seed=int(seed),
            use_gpu_compute=_use_gpu,
            gpu_all_stages=_all_gpu,
            shader_dir=_SHADER_DIR,
        )

        if blocking:
            _DRAIN_KEYS = ("kind", "tag", "pos", "dir", "seg_start", "path_len",
                           "bounce", "is_sensor", "hit_tri", "mat_idx", "color_flag",
                           "hit_group_id", "amp_re", "amp_im")
            acc: Dict[str, list] = {k: [] for k in _DRAIN_KEYS}
            while True:
                recs = self.tracer.drain_records(max_n=100_000)
                if len(recs.get("kind", [])) > 0:
                    for k in _DRAIN_KEYS:
                        if k in recs:
                            acc[k].append(np.asarray(recs[k]))
                if self.tracer.in_flight_count() == 0:
                    recs = self.tracer.drain_records(max_n=500_000)
                    if len(recs.get("kind", [])) > 0:
                        for k in _DRAIN_KEYS:
                            if k in recs:
                                acc[k].append(np.asarray(recs[k]))
                    break
                time.sleep(0.001)
            return {k: np.concatenate(v) if v else np.array([]) for k, v in acc.items()}

        self._ensure_drain_loop()
        self._async_forward_launched_count += int(total)
        return total

    def trace_sensor_cast(
        self,
        rays_per_sensor: int,
        seed: int,
        max_bounces: int = 6,
    ) -> int:
        """Back-cast a requested budget of sensor samples through the lens.

        ``rays_per_sensor`` is treated as the total reverse-ray budget for this
        call.  Samples are distributed over all valid sensor pixels with
        sub-pixel jitter and independent aperture samples, so increasing the
        count increases film/aperture coverage instead of multiplying a fixed
        square launch lattice.

        NON-STANDARD BDPT NOTE:
        A normal BDPT camera subpath samples a continuous film coordinate and
        lens sample, stores the camera PDF/throughput, and later splats through
        a reconstruction filter.  This path now preserves continuous film UV
        in the ray tag and endpoint cache, records launch-domain camera PDFs,
        and feeds a provisional endpoint PDF into the connector.  Full lens
        Jacobians and multi-strategy MIS are still pending.
        """
        plate   = self.scene.image_plate
        res     = int(max(4, plate.sensor_res))
        # The mesh build and optics setup may update the assembly after the
        # scene fields are cloned.  Refresh the real reverse-ray target here so
        # sensor-cast launches defer to the assembly's current optical extents.
        _asm = getattr(self, "_lens_assembly", None)
        if _asm is not None:
            try:
                _ap_cen, _ap_r = _asm.backward_ray_target()
                if _ap_cen is not None and float(_ap_r) > 0.0:
                    self.aperture_centroid = np.asarray(_ap_cen, dtype=np.float64)
                    self.aperture_radius = float(_ap_r)
            except Exception:
                pass
        if self.aperture_radius <= 0.0:
            self._async_backward_skip_reason = "aperture_radius<=0"
            print(
                "[sensor-cast-skip]",
                self._async_backward_skip_reason,
                f"aperture_radius={float(self.aperture_radius):.6g}",
                flush=True,
            )
            return 0
        ap_r = float(self.aperture_radius)

        # ── Build circular pixel grid ────────────────────────────────────────
        # Regular UV cell centres in [-1, 1], clipped to unit disc.
        u   = np.linspace(-1.0, 1.0, res + 1)
        um  = 0.5 * (u[:-1] + u[1:])          # cell centres
        gy, gz = np.meshgrid(um, um, indexing='ij')  # (res, res)
        gy  = gy.ravel()
        gz  = gz.ravel()
        in_disc = (gy ** 2 + gz ** 2) <= 1.0
        disc_grid_indices = np.where(in_disc)[0]  # row*res+col for each disc pixel
        n_pixels = int(disc_grid_indices.shape[0])
        if n_pixels == 0:
            self._async_backward_skip_reason = "sensor_disc_pixels=0"
            print("[sensor-cast-skip]", self._async_backward_skip_reason, flush=True)
            return 0

        plate_x = float(plate.x)
        n_sensor_channels = 3
        full_sensor_min = int(n_pixels * n_sensor_channels)
        if rays_per_sensor > 0:
            total = int(max(full_sensor_min, rays_per_sensor))
        else:
            total = int(max(full_sensor_min, n_pixels * int(max(1, plate.bokeh_rays)) * n_sensor_channels))
        self._async_backward_attempt_count += int(total)

        # ── Orthonormal basis for aperture disc ──────────────────────────────
        ap_n = self.aperture_normal / (np.linalg.norm(self.aperture_normal) + 1e-30)
        tb   = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(ap_n, tb))) > 0.9:
            tb = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tb -= np.dot(tb, ap_n) * ap_n;  tb /= np.linalg.norm(tb) + 1e-30
        tc   = np.cross(ap_n, tb)

        rng = np.random.default_rng(seed)
        sample_i = np.arange(total, dtype=np.int64)
        channel_idx = (sample_i % n_sensor_channels).astype(np.int32)

        # Deterministic raster scan across the disc pixel set with sub-pixel jitter.
        pix_offset = self._sweep_pixel_offset % max(1, n_pixels)
        self._sweep_pixel_offset += total // n_sensor_channels
        pixel_seq = disc_grid_indices[(pix_offset + (sample_i // n_sensor_channels)) % n_pixels].astype(np.int64)
        pix_y = pixel_seq // res
        pix_z = pixel_seq % res
        jy = rng.random(total)
        jz = rng.random(total)
        y_norm = ((pix_y.astype(np.float64) + jy) / float(res)) * 2.0 - 1.0
        z_norm = ((pix_z.astype(np.float64) + jz) / float(res)) * 2.0 - 1.0
        rr = np.sqrt(y_norm * y_norm + z_norm * z_norm)
        outside = rr > 1.0
        if np.any(outside):
            scale = 0.999999 / np.maximum(rr[outside], 1.0e-30)
            y_norm[outside] *= scale
            z_norm[outside] *= scale
        film_uv = np.column_stack([
            0.5 * (y_norm + 1.0),
            0.5 * (z_norm + 1.0),
        ]).astype(np.float32)

        origins = np.stack(
            [
                np.full(total, plate_x, dtype=np.float64),
                y_norm * float(plate.radius),
                z_norm * float(plate.radius),
            ],
            axis=1,
        )

        # Independent full-aperture sampling for every ray; the count covers
        # the complete exit-pupil disc instead of a tiny stencil.
        eff_ap_r = ap_r
        ap_ang = rng.uniform(0.0, 2.0 * math.pi, total)
        ap_rad = np.sqrt(rng.uniform(0.0, 1.0, total)) * eff_ap_r
        s_cx = ap_rad * np.cos(ap_ang)
        s_cy = ap_rad * np.sin(ap_ang)

        ap_pts = (self.aperture_centroid
                  + s_cx[:, None] * tb[None, :]
                  + s_cy[:, None] * tc[None, :])  # (total, 3)

        # ── Assemble origins / directions ─────────────────────────────────────
        d          = ap_pts - origins
        nrm        = np.linalg.norm(d, axis=1, keepdims=True)
        directions = d / np.maximum(nrm, 1e-30)
        src_ids    = pixel_seq.astype(np.int32, copy=False)
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
                f"ray_budget={total}",
                flush=True,
            )
            self._sensor_aim_reported = True

        # Reverse paths are launched through RGB sensor sensitivity lobes.  This
        # makes the sensor an RGB spectral emitter instead of a flat white source.
        amp_scale = float(self.sensor_amp_gain)
        sens_rgb = _sensor_rgb_sensitivity_bands(self.freq_hz[:self.n_bands])
        shutter_w = film_shutter_transmission(plate, film_uv)
        if not np.any(shutter_w > 0.0):
            self._async_backward_skip_reason = "shutter_closed"
            print(
                "[sensor-cast-skip]",
                self._async_backward_skip_reason,
                f"mode={getattr(plate, 'shutter_mode', 'open')}",
                f"open={float(getattr(plate, 'shutter_open', 1.0)):.3f}",
                flush=True,
            )
            return 0
        sensor_amps = (amp_scale * shutter_w[:, None] * sens_rgb[channel_idx]).astype(np.complex128, copy=True)
        live = shutter_w > 0.0
        if not np.all(live):
            origins = origins[live]
            directions = directions[live]
            ap_pts = ap_pts[live]
            src_ids = src_ids[live]
            cflag_arr = cflag_arr[live]
            channel_idx = channel_idx[live]
            shutter_w = shutter_w[live]
            sensor_amps = sensor_amps[live]
            film_uv = film_uv[live]
            total = int(origins.shape[0])
        tag_arr = _pack_bdpt_film_tag(channel_idx, film_uv)
        camera_batch_id = int(self._camera_sample_batch_id)
        self._camera_sample_batch_id = int((self._camera_sample_batch_id + 1) & 0xFFFFFFFF)
        film_area_pdf = 1.0 / max(math.pi * float(plate.radius) * float(plate.radius), 1.0e-30)
        aperture_area_pdf = 1.0 / max(math.pi * eff_ap_r * eff_ap_r, 1.0e-30)
        channel_pdf = 1.0 / float(max(1, n_sensor_channels))
        self._last_backward_film_pdf = float(film_area_pdf)
        self._last_backward_aperture_pdf = float(aperture_area_pdf)
        self._last_backward_channel_pdf = float(channel_pdf)
        self._last_backward_strategy_pdf = float(film_area_pdf) * float(aperture_area_pdf) * float(channel_pdf)
        cam_film_uv = np.asarray(film_uv, dtype=np.float32).copy()
        cam_origins = np.asarray(origins, dtype=np.float64).copy()
        cam_ap_pts = np.asarray(ap_pts, dtype=np.float64).copy()
        cam_directions = np.asarray(directions, dtype=np.float64).copy()
        cam_channel_idx = np.asarray(channel_idx, dtype=np.int32).copy()
        cam_shutter_w = np.asarray(shutter_w, dtype=np.float64).copy()
        cam_sensor_weights = np.asarray(sens_rgb[channel_idx], dtype=np.float32).copy()
        cam_jac = np.zeros(total, dtype=np.float32)
        cam_solid_pdf = np.zeros(total, dtype=np.float32)
        cam_phase_jac = np.zeros(total, dtype=np.float32)
        cam_phase_pdf = np.zeros(total, dtype=np.float32)

        if _asm is not None and getattr(_asm, "optics", None) is not None:
            detailed = _asm.evaluate_transfer_detailed(RayBundle(origins, directions))
            self._append_optical_transfer_events(detailed, film_uv, stream=1, batch_id=camera_batch_id)
            result = detailed.result
            passed = result.status == int(TerminationReason.PASSED.value)
            n_passed = int(np.count_nonzero(passed))
            n_absorbed = int(total) - n_passed
            self._async_backward_parametric_absorbed_count += max(0, n_absorbed)
            if n_passed <= 0:
                self._async_backward_skip_reason = "parametric_no_transmit"
                print(
                    "[sensor-cast-parametric]",
                    f"requested={int(total):_}",
                    f"evaluated={int(total):_}",
                    "passed=0",
                    f"absorbed={n_absorbed:_}",
                    f"attempted_total={self._async_backward_attempt_count:_}",
                    f"absorbed_total={self._async_backward_parametric_absorbed_count:_}",
                    f"opt_fail={self.bdpt_last_optical_transfer.get('top_fail_element', -1)}:{self.bdpt_last_optical_transfer.get('top_fail_reason', '')}",
                    flush=True,
                )
                self._append_camera_sample_records(
                    batch_id=camera_batch_id,
                    film_uv=cam_film_uv,
                    film_pos=cam_origins,
                    aperture_pos=cam_ap_pts,
                    directions=cam_directions,
                    channel_idx=cam_channel_idx,
                    shutter_weight=cam_shutter_w,
                    sensor_weights=cam_sensor_weights,
                    film_area_pdf=film_area_pdf,
                    aperture_area_pdf=aperture_area_pdf,
                    channel_pdf=channel_pdf,
                    aperture_to_solid_angle_jac=cam_jac,
                    solid_angle_pdf=cam_solid_pdf,
                    phase_space_jac=cam_phase_jac,
                    phase_space_pdf=cam_phase_pdf,
                )
                return 0
            jac_passed, phase_jac_passed = self._estimate_camera_transfer_jacobians(
                _asm,
                cam_origins[passed],
                cam_ap_pts[passed],
                np.asarray(result.origins[passed], dtype=np.float64),
                np.asarray(result.directions[passed], dtype=np.float64),
                plate_radius=float(plate.radius),
                tb=tb,
                tc=tc,
                aperture_radius=eff_ap_r,
            )
            jac_safe = np.maximum(jac_passed, 1.0e-12)
            bwd_launch_pdf = (float(aperture_area_pdf) / jac_safe).astype(np.float32)
            phase_safe = np.maximum(phase_jac_passed, 1.0e-20)
            bwd_phase_pdf = (float(film_area_pdf) * float(aperture_area_pdf) / phase_safe).astype(np.float32)
            cam_jac[passed] = jac_passed.astype(np.float32, copy=False)
            cam_solid_pdf[passed] = bwd_launch_pdf
            cam_phase_jac[passed] = phase_jac_passed.astype(np.float32, copy=False)
            cam_phase_pdf[passed] = bwd_phase_pdf
            exit_cos = np.clip(np.abs(np.asarray(result.directions[passed], dtype=np.float64)[:, 0]), 1.0e-4, 1.0)
            bwd_area_pdf = np.full(n_passed, float(film_area_pdf) * float(aperture_area_pdf), dtype=np.float32)
            self._last_backward_cos_exit = float(np.mean(exit_cos)) if exit_cos.size else 1.0
            for _tag, _launch_pdf, _area_pdf, _jac in zip(tag_arr[passed], bwd_launch_pdf, bwd_area_pdf, jac_passed):
                _key = int(np.uint64(_tag))
                self._backward_launch_pdf_by_tag[_key] = float(_launch_pdf)
                self._backward_area_pdf_by_tag[_key] = float(_area_pdf)
                self._backward_jacobian_by_tag[_key] = float(_jac)
            origins = np.asarray(result.origins[passed], dtype=np.float64)
            directions = np.asarray(result.directions[passed], dtype=np.float64)
            origins = origins + directions * 1.0e-6
            src_ids = src_ids[passed]
            tag_arr = tag_arr[passed]
            cflag_arr = cflag_arr[passed]
            sensor_amps = sensor_amps[passed]
            film_uv = film_uv[passed]
            total = n_passed
            print(
                "[sensor-cast-parametric]",
                f"requested={int(passed.shape[0]):_}",
                f"evaluated={int(passed.shape[0]):_}",
                f"passed={n_passed:_}",
                f"absorbed={n_absorbed:_}",
                f"attempted_total={self._async_backward_attempt_count:_}",
                f"absorbed_total={self._async_backward_parametric_absorbed_count:_}",
                f"opt_fail={self.bdpt_last_optical_transfer.get('top_fail_element', -1)}:{self.bdpt_last_optical_transfer.get('top_fail_reason', '')}",
                flush=True,
            )

        self._append_camera_sample_records(
            batch_id=camera_batch_id,
            film_uv=cam_film_uv,
            film_pos=cam_origins,
            aperture_pos=cam_ap_pts,
            directions=cam_directions,
            channel_idx=cam_channel_idx,
            shutter_weight=cam_shutter_w,
            sensor_weights=cam_sensor_weights,
            film_area_pdf=film_area_pdf,
            aperture_area_pdf=aperture_area_pdf,
            channel_pdf=channel_pdf,
            aperture_to_solid_angle_jac=cam_jac,
            solid_angle_pdf=cam_solid_pdf,
            phase_space_jac=cam_phase_jac,
            phase_space_pdf=cam_phase_pdf,
        )

        _use_gpu = self.compute_mode in ("gpu", "mixed")
        _all_gpu = self.compute_mode == "gpu"
        if self.cull_infinite_rays:
            origins, directions, sensor_amps, src_ids, tag_arr, cflag_arr = _cull_finite_rays(
                origins, directions, sensor_amps, src_ids, tag_arr, cflag_arr, label="bwd")

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
        self._async_backward_launched_count += int(total)
        self._async_backward_skip_reason = ""
        return total

    def _fast_bdpt_feed(self, records: dict) -> None:
        """Feed BDPT endpoints immediately from raw drained records.

        Called as the very first operation on every drain batch so BDPT
        accumulates endpoints with zero pipeline delay.
        """
        self._append_bdpt_segment_records(records)
        kinds   = np.asarray(records["kind"], dtype=np.uint8)
        strike  = kinds == 0
        if not np.any(strike):
            return
        cflags    = np.asarray(records["color_flag"], dtype=np.uint8)
        stream_ok = strike & ((cflags == 0) | (cflags == 1))
        if not np.any(stream_ok):
            return
        amp_re  = np.asarray(records["amp_re"])
        amp_im  = np.asarray(records["amp_im"])
        pos     = np.asarray(records["pos"])
        bounces = np.asarray(records["bounce"])
        tags_r  = records.get("tag")
        tags    = np.asarray(tags_r, dtype=np.uint64) if tags_r is not None \
                  else np.zeros(kinds.shape[0], dtype=np.uint64)
        _film_tag_valid, film_uv = _unpack_bdpt_film_tag(tags)
        gid_r   = records.get("hit_group_id")
        gid     = np.asarray(gid_r, dtype=np.int32) if gid_r is not None \
                  else np.full(kinds.shape[0], -1, dtype=np.int32)
        htri_r  = records.get("hit_tri")
        htri    = np.asarray(htri_r, dtype=np.int32) if htri_r is not None \
                  else np.full(kinds.shape[0], -1, dtype=np.int32)
        nb = min(int(amp_re.shape[1]), int(self.n_bands))
        self._append_async_bdpt_records(
            vm_kinds   = kinds[stream_ok],
            vm_cflags  = cflags[stream_ok].astype(np.int32),
            vm_pos     = pos[stream_ok],
            vm_re      = amp_re[stream_ok, :nb],
            vm_im      = amp_im[stream_ok, :nb],
            vm_bounces = bounces[stream_ok],
            vm_tags    = tags[stream_ok],
            vm_gid     = gid[stream_ok],
            vm_hit_tri = htri[stream_ok],
            vm_film_uv = film_uv[stream_ok],
        )

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
        _min_batch =   4_096
        _max_batch = 1_000_000
        # Throughput tracking for adaptive batch sizing.
        # The target drain window is an alpha-weighted mix of:
        #   - a base 20 ms constant (pipeline-driven)
        #   - a fraction (50%) of the per-frame budget at the preferred FPS
        # fps_alpha steers how strongly the frame-rate preference matters.
        _BASE_DRAIN_S    = 0.020
        _FRAME_FRACTION  = 0.50   # drain may use up to this share of one frame
        _throughput_est  = 100_000_000.0  # start high → first call requests full _max_batch
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
                    self._fast_bdpt_feed(records)
                    self._accumulate_records(records)
                    # Update rolling throughput estimate every 50ms or 1k records.
                    _n_since_update += n
                    t_now = time.perf_counter()
                    dt    = t_now - _t_last
                    if dt >= 0.050 or _n_since_update >= 1_000:
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

        # Extract per-hit group IDs (set in T1 from tri_param_group_of_tri).
        raw_hgid = records.get("hit_group_id", None)
        if raw_hgid is not None:
            vm_gid = np.asarray(raw_hgid)[vis_mask].astype(np.int32)
        else:
            vm_gid = np.full(int(np.count_nonzero(vis_mask)), -1, dtype=np.int32)

        # Extract hit triangle indices for forward-preview filtering.
        raw_htri = records.get("hit_tri", None)
        if raw_htri is not None:
            vm_hit_tri = np.asarray(raw_htri)[vis_mask].astype(np.int32)
        else:
            vm_hit_tri = np.full(int(np.count_nonzero(vis_mask)), -1, dtype=np.int32)

        fwd_strike_for_count = (vm_kinds == 0) & (vm_cflags == 0) & (vm_hit_tri >= 0)
        if np.any(fwd_strike_for_count):
            tri_kind = getattr(self, "tri_kind", None)
            self._async_forward_strike_count += int(np.count_nonzero(fwd_strike_for_count))
            if tri_kind is not None and len(tri_kind) > 0:
                safe_htri = np.clip(vm_hit_tri[fwd_strike_for_count], 0, len(tri_kind) - 1)
                lens_hit = np.asarray(tri_kind[safe_htri]) == TRI_KIND_LENS
                n_lens = int(np.count_nonzero(lens_hit))
                self._async_forward_lens_hit_count += n_lens
                if n_lens > 0:
                    fwd_bounces = np.asarray(bounces[vis_mask], dtype=np.int32)[fwd_strike_for_count]
                    self._async_forward_lens_hit_after_bounce_count += int(np.count_nonzero(lens_hit & (fwd_bounces > 0)))

        bwd_strike_for_count = (vm_kinds == 0) & (vm_cflags == 1) & (vm_hit_tri >= 0)
        if np.any(bwd_strike_for_count):
            self._async_backward_strike_count += int(np.count_nonzero(bwd_strike_for_count))

        # For class-3 also count the ray origin (image-plate surface) as a hit.
        rev_strike_mask = (disp_class == 3)
        if np.any(rev_strike_mask):
            vm_pos_all   = np.concatenate([vm_pos,   vm_ss[rev_strike_mask]],    axis=0)
            amp_v_all    = np.concatenate([amp_v,    amp_v[rev_strike_mask]],    axis=0)
            disp_cls_all = np.concatenate([disp_class, np.full(int(np.count_nonzero(rev_strike_mask)), 3, dtype=np.int32)], axis=0)
            gid_all      = np.concatenate([vm_gid,   vm_gid[rev_strike_mask]],   axis=0)
        else:
            vm_pos_all   = vm_pos
            amp_v_all    = amp_v
            disp_cls_all = disp_class
            gid_all      = vm_gid

        def _records_to_preview(mask: np.ndarray, dst: np.ndarray) -> None:
            if not np.any(mask):
                return
            # Use scene view_radius for Y-Z extent: scene-side hits are spread
            # over the full scene volume, not just the millimeter-scale sensor disc.
            vr = float(max(1.0e-9, self.scene.view_radius))
            p = vm_pos[mask]
            in_view = (np.abs(p[:, 1]) <= vr) & (np.abs(p[:, 2]) <= vr)
            # Exclude lens-glass and camera-optics hits: they dominate hit counts
            # and obscure actual scene geometry in the forward preview.
            htri_m = vm_hit_tri[mask]
            tri_kind = getattr(self, "tri_kind", None)
            if tri_kind is not None and htri_m.size > 0:
                safe = np.clip(htri_m, 0, len(tri_kind) - 1)
                kind_m = np.where(htri_m >= 0, tri_kind[safe], TRI_KIND_DEFAULT)
                is_optic = (kind_m == TRI_KIND_LENS) | (kind_m == TRI_KIND_APERTURE) | (kind_m == TRI_KIND_SENSOR)
                m_img = in_view & ~is_optic
            else:
                m_img = in_view
            if np.any(m_img):
                p_img = p[m_img]
                re_img = vm_re[mask][m_img]
                im_img = vm_im[mask][m_img]
                nb_img = min(int(re_img.shape[1]), self.n_bands)
                if nb_img > 0:
                    mag = np.sqrt(np.maximum(0.0, re_img[:, :nb_img] ** 2 + im_img[:, :nb_img] ** 2))
                    wl_nm = (C_LIGHT / np.maximum(np.asarray(self.freq_hz[:nb_img], dtype=np.float64), EPS)) * 1.0e9
                    rgb_w = _wavelength_to_rgb_weights(wl_nm).astype(np.float32)
                    rgb = np.einsum("nb,bc->nc", mag, rgb_w[:nb_img], optimize=True).astype(np.float32)
                else:
                    rgb = np.repeat(amp_v[mask][m_img, None], 3, axis=1).astype(np.float32)
                res = int(self._forward_img_res)
                iy = np.clip(((p_img[:, 1] + vr) / (2.0 * vr) * res).astype(np.int32), 0, res - 1)
                iz = np.clip(((p_img[:, 2] + vr) / (2.0 * vr) * res).astype(np.int32), 0, res - 1)
                flat_idx = (iy * res + iz).astype(np.int64)
                with self._segs_lock:
                    for ch in range(3):
                        dst[:, :, ch] += np.bincount(
                            flat_idx, weights=rgb[:, ch].astype(np.float64),
                            minlength=res * res,
                        ).reshape(res, res).astype(np.float32)

        # Pure preview feeds: project all strike records into the image-plane Y/Z
        # grid.  Do not require the ray to hit the image plate; this is a live
        # diagnostic of the forward/reverse ray distributions before final image
        # formation rules are trusted.
        fwd_strike = (vm_kinds == 0) & (vm_cflags == 0)
        rev_strike = (vm_kinds == 0) & (vm_cflags == 1)
        _records_to_preview(fwd_strike, self._forward_img_accum)
        _records_to_preview(rev_strike, self._reverse_img_accum)

        # Pack raw hit positions into ring-buffer rows: (x, y, z, amp, class, gid)
        pts = np.empty((vm_pos_all.shape[0], 6), dtype=np.float32)
        pts[:, :3] = vm_pos_all
        pts[:, 3]  = amp_v_all
        pts[:, 4]  = disp_cls_all.astype(np.float32)
        pts[:, 5]  = gid_all.astype(np.float32)
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

    def _append_async_bdpt_records(
        self,
        *,
        vm_kinds: np.ndarray,
        vm_cflags: np.ndarray,
        vm_pos: np.ndarray,
        vm_re: np.ndarray,
        vm_im: np.ndarray,
        vm_bounces: np.ndarray,
        vm_tags: np.ndarray,
        vm_gid: np.ndarray,
        vm_hit_tri: np.ndarray,
        vm_film_uv: Optional[np.ndarray] = None,
    ) -> None:
        """Retain async pipeline strike records as BDPT half-path endpoints."""
        if vm_pos.shape[0] == 0:
            return
        strike = np.asarray(vm_kinds) == 0
        stream_ok = (np.asarray(vm_cflags) == 0) | (np.asarray(vm_cflags) == 1)
        base_mask = strike & stream_ok
        # Only keep endpoints on diffuse/structural surfaces (TRI_KIND_DEFAULT).
        # Lens glass (TRI_KIND_LENS), aperture stops, emitters, and sensors produce
        # endpoints where shadow rays are immediately blocked by the same surface,
        # causing every connection to test as occluded → black BDPT image.
        tri_kind = getattr(self, "tri_kind", None)
        if tri_kind is not None and tri_kind.shape[0] > 0:
            htri = np.asarray(vm_hit_tri, dtype=np.int32)
            htri_clamped = np.clip(htri, 0, tri_kind.shape[0] - 1)
            kind_at_hit = np.asarray(tri_kind[htri_clamped], dtype=np.int8)
            # htri == -1 means no hit recorded; treat as non-default → exclude
            scene_surface = (htri >= 0) & (kind_at_hit == TRI_KIND_DEFAULT)
            base_mask = base_mask & scene_surface
        if not np.any(base_mask):
            return

        idx = np.nonzero(base_mask)[0]
        nb = min(int(vm_re.shape[1]), int(self.n_bands))
        if nb <= 0:
            return

        amp_re_sel = np.asarray(vm_re[idx, :nb], dtype=np.float32)
        amp_im_sel = np.asarray(vm_im[idx, :nb], dtype=np.float32)
        amp_sq = amp_re_sel * amp_re_sel + amp_im_sel * amp_im_sel
        base_cflags = np.asarray(vm_cflags[idx], dtype=np.int32)
        base_is_bwd = base_cflags == 1
        thresh = np.where(
            base_is_bwd,
            float(max(0.0, self.sensor_min_amplitude)),
            float(max(0.0, self._min_amplitude)),
        ).astype(np.float32)
        active = amp_sq > (thresh[:, None] * thresh[:, None])
        ev_rel, band_ids = np.nonzero(active)
        if ev_rel.size == 0:
            return

        event_idx = idx[ev_rel]
        base_subpath = np.empty(idx.shape[0], dtype=np.uint32)

        pixel_mask = np.uint64((1 << 60) - 1)
        base_subpath[base_is_bwd] = (np.asarray(vm_tags[idx[base_is_bwd]], dtype=np.uint64) & pixel_mask).astype(np.uint32)
        n_fwd_events = int(np.count_nonzero(~base_is_bwd))
        next_forward = int(self._async_bdpt_next_subpath)
        if n_fwd_events > 0:
            base_subpath[~base_is_bwd] = np.arange(
                next_forward,
                next_forward + n_fwd_events,
                dtype=np.uint32,
            )
        self._async_bdpt_next_subpath = next_forward + n_fwd_events

        n_new = int(ev_rel.size)
        new = np.empty(n_new, dtype=BDPT_ENDPOINT_DTYPE)
        new["id"] = base_subpath[ev_rel].astype(np.uint32, copy=False)
        new["band"] = band_ids.astype(np.uint32, copy=False)
        new["pos"] = np.asarray(vm_pos[event_idx], dtype=np.float32)
        new["amp"] = (
            amp_re_sel[ev_rel, band_ids].astype(np.float32, copy=False)
            + 1j * amp_im_sel[ev_rel, band_ids].astype(np.float32, copy=False)
        ).astype(np.complex64, copy=False)
        new["stream"] = np.where(base_is_bwd[ev_rel], 1, 0).astype(np.uint8, copy=False)
        new["film_uv"] = 0.0
        if vm_film_uv is not None:
            film_uv = np.asarray(vm_film_uv, dtype=np.float32)
            if film_uv.ndim == 2 and film_uv.shape[0] == vm_pos.shape[0] and film_uv.shape[1] >= 2:
                new["film_uv"] = np.clip(film_uv[event_idx, :2], 0.0, 1.0)
        tag_event = np.asarray(vm_tags[event_idx], dtype=np.uint64)
        bwd_lookup = np.fromiter(
            (self._backward_launch_pdf_by_tag.get(int(t), self._last_backward_aperture_pdf) for t in tag_event),
            dtype=np.float32,
            count=int(tag_event.shape[0]),
        )
        new["launch_pdf"] = np.where(
            base_is_bwd[ev_rel],
            bwd_lookup,
            np.float32(self._emitter_launch_pdf),
        )

        self._bdpt_endpoints.append(new)

    def _append_bdpt_segment_records(self, records: dict) -> None:
        """Retain ray subsegments in a MIS-facing structured format.

        This records path geometry and provisional launch PDFs.  Backward
        camera segments receive the per-ray camera endpoint PDF carried by tag;
        full reverse PDFs, lens Jacobians, and material sampling PDFs remain
        explicitly absent.
        """
        if not records:
            return
        kinds = np.asarray(records.get("kind", []), dtype=np.uint8)
        if kinds.size == 0:
            return
        cflags = np.asarray(records.get("color_flag", np.zeros(kinds.shape[0], dtype=np.uint8)), dtype=np.uint8)
        keep = ((cflags == 0) | (cflags == 1)) & ((kinds == 0) | (kinds == 3))
        if not np.any(keep):
            return

        seg_start = np.asarray(records.get("seg_start"), dtype=np.float32)
        pos = np.asarray(records.get("pos"), dtype=np.float32)
        dirs = np.asarray(records.get("dir", pos - seg_start), dtype=np.float32)
        amp_re = np.asarray(records.get("amp_re"), dtype=np.float32)
        amp_im = np.asarray(records.get("amp_im"), dtype=np.float32)
        if seg_start.ndim != 2 or pos.ndim != 2 or amp_re.ndim != 2 or amp_im.ndim != 2:
            return

        idx = np.nonzero(keep)[0]
        nb = min(int(amp_re.shape[1]), int(amp_im.shape[1]), int(self.n_bands))
        if nb <= 0:
            return
        re_sel = amp_re[idx, :nb]
        im_sel = amp_im[idx, :nb]
        amp_sq = re_sel * re_sel + im_sel * im_sel
        thresh = np.where(
            cflags[idx] == 1,
            float(max(0.0, self.sensor_min_amplitude)),
            float(max(0.0, self._min_amplitude)),
        ).astype(np.float32)
        active = amp_sq > (thresh[:, None] * thresh[:, None])
        ev_rel, band_ids = np.nonzero(active)
        if ev_rel.size == 0:
            return

        event_idx = idx[ev_rel]
        tags_r = records.get("tag")
        tags = np.asarray(tags_r, dtype=np.uint64) if tags_r is not None else np.zeros(kinds.shape[0], dtype=np.uint64)
        _film_valid, film_uv = _unpack_bdpt_film_tag(tags)
        bounces = np.asarray(records.get("bounce", np.zeros(kinds.shape[0], dtype=np.int32)), dtype=np.int32)
        path_len = np.asarray(records.get("path_len", np.zeros(kinds.shape[0], dtype=np.float32)), dtype=np.float32)
        gid = np.asarray(records.get("hit_group_id", np.full(kinds.shape[0], -1, dtype=np.int32)), dtype=np.int32)
        htri = np.asarray(records.get("hit_tri", np.full(kinds.shape[0], -1, dtype=np.int32)), dtype=np.int32)

        base_ids = np.empty(idx.shape[0], dtype=np.uint32)
        is_bwd = cflags[idx] == 1
        if np.any(is_bwd):
            base_ids[is_bwd] = (tags[idx[is_bwd]] & np.uint64((1 << 29) - 1)).astype(np.uint32)
        n_fwd = int(np.count_nonzero(~is_bwd))
        if n_fwd > 0:
            start = int(self._async_bdpt_next_segment_subpath)
            base_ids[~is_bwd] = np.arange(start, start + n_fwd, dtype=np.uint32)
            self._async_bdpt_next_segment_subpath = start + n_fwd

        rows = np.empty(int(ev_rel.size), dtype=BDPT_SEGMENT_DTYPE)
        rows["subpath_id"] = base_ids[ev_rel]
        rows["band"] = band_ids.astype(np.uint32, copy=False)
        rows["p0"] = seg_start[event_idx]
        rows["p1"] = pos[event_idx]
        rows["dir"] = dirs[event_idx]
        rows["amp"] = (
            re_sel[ev_rel, band_ids].astype(np.float32, copy=False)
            + 1j * im_sel[ev_rel, band_ids].astype(np.float32, copy=False)
        ).astype(np.complex64, copy=False)
        rows["film_uv"] = np.where(
            (cflags[event_idx] == 1)[:, None],
            np.clip(film_uv[event_idx], 0.0, 1.0),
            np.zeros((event_idx.shape[0], 2), dtype=np.float32),
        )
        rows["path_len"] = path_len[event_idx].astype(np.float32, copy=False)
        is_bwd_ev = cflags[event_idx] == 1
        tag_event = np.asarray(tags[event_idx], dtype=np.uint64)
        bwd_strat = np.fromiter(
            (self._backward_launch_pdf_by_tag.get(int(t), self._last_backward_aperture_pdf) for t in tag_event),
            dtype=np.float32,
            count=int(tag_event.shape[0]),
        )
        bwd_area = np.fromiter(
            (self._backward_area_pdf_by_tag.get(
                int(t),
                self._last_backward_film_pdf * self._last_backward_aperture_pdf,
            ) for t in tag_event),
            dtype=np.float32,
            count=int(tag_event.shape[0]),
        )
        bwd_jac = np.fromiter(
            (self._backward_jacobian_by_tag.get(int(t), 0.0) for t in tag_event),
            dtype=np.float32,
            count=int(tag_event.shape[0]),
        )
        rows["pdf_fwd"]        = np.where(is_bwd_ev, bwd_strat, np.float32(0.0))
        rows["pdf_rev"]        = np.float32(0.0)
        rows["pdf_area"]       = np.where(is_bwd_ev, bwd_area, np.float32(0.0))
        rows["pdf_solid_angle"] = np.where(is_bwd_ev, bwd_strat, np.float32(0.0))
        rows["strategy_id"]   = np.where(is_bwd_ev, np.uint16(CAMERA_STRATEGY_SENSOR_APERTURE), np.uint16(0))
        rows["sample_domain"] = np.where(
            is_bwd_ev, BDPT_SAMPLE_DOMAIN_FILM_APERTURE, BDPT_SAMPLE_DOMAIN_HEMISPHERE,
        ).astype(np.uint8)
        rows["stream"] = np.where(cflags[event_idx] == 1, 1, 0).astype(np.uint8, copy=False)
        rows["kind"] = kinds[event_idx].astype(np.uint8, copy=False)
        rows["bounce"] = bounces[event_idx].astype(np.int32, copy=False)
        rows["hit_group_id"] = gid[event_idx].astype(np.int32, copy=False)
        rows["hit_tri"] = htri[event_idx].astype(np.int32, copy=False)
        rows["flags"] = np.where(
            is_bwd_ev & (bwd_jac <= 0.0),
            BDPT_SEGMENT_FLAG_JACOBIAN_MISSING,
            np.uint32(0),
        )
        self._bdpt_segments.append(rows)

    def _estimate_camera_transfer_jacobians(
        self,
        assembly,
        film_origins: np.ndarray,
        aperture_points: np.ndarray,
        base_exit_origins: np.ndarray,
        base_exit_dirs: np.ndarray,
        *,
        plate_radius: float,
        tb: np.ndarray,
        tc: np.ndarray,
        aperture_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference camera transfer Jacobians for backward rays.

        Returns:
          1. |dω_exit / dA_aperture| with film fixed.
          2. |d(exit_y, exit_z, dir_y, dir_z) /
              d(film_y, film_z, aperture_y, aperture_z)|.

        The second determinant is the full first-order geometric-optics
        transfer density for this camera sample space.  It is computed by
        retracing four neighboring samples through the same parametric assembly.
        """
        origins = np.asarray(film_origins, dtype=np.float64)
        ap = np.asarray(aperture_points, dtype=np.float64)
        base_o = np.asarray(base_exit_origins, dtype=np.float64)
        base = np.asarray(base_exit_dirs, dtype=np.float64)
        n = int(origins.shape[0])
        out_ap = np.zeros(n, dtype=np.float32)
        out_phase = np.zeros(n, dtype=np.float32)
        if n == 0 or assembly is None:
            return out_ap, out_phase
        eps_ap = float(max(1.0e-7, min(1.0e-4, abs(float(aperture_radius)) * 1.0e-4)))
        eps_film = float(max(1.0e-7, min(1.0e-4, abs(float(plate_radius)) * 1.0e-4)))
        tb_v = np.asarray(tb, dtype=np.float64)
        tc_v = np.asarray(tc, dtype=np.float64)
        native = getattr(_sk, "estimate_lens_camera_jacobians", None)
        if native is not None:
            try:
                payload = np.ascontiguousarray(assembly.build_parametric_payload(), dtype=np.float32)
                aj, pj = native(
                    payload,
                    np.ascontiguousarray(origins, dtype=np.float64),
                    np.ascontiguousarray(ap, dtype=np.float64),
                    np.ascontiguousarray(base_o, dtype=np.float64),
                    np.ascontiguousarray(base, dtype=np.float64),
                    float(plate_radius),
                    float(aperture_radius),
                    np.ascontiguousarray(tb_v, dtype=np.float64),
                    np.ascontiguousarray(tc_v, dtype=np.float64),
                    0,
                )
                return (
                    np.asarray(aj, dtype=np.float32),
                    np.asarray(pj, dtype=np.float32),
                )
            except Exception as exc:
                if not getattr(self, "_lens_jacobian_native_failed_reported", False):
                    print(f"[lens-jacobian] native C++ path unavailable, using Python fallback: {exc}", flush=True)
                    self._lens_jacobian_native_failed_reported = True

        def _trace_variant(ori: np.ndarray, apt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            dirs = apt - ori
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1.0e-30)
            res = assembly.evaluate_transfer(RayBundle(ori, dirs))
            ok = res.status == int(TerminationReason.PASSED.value)
            return np.asarray(res.origins, dtype=np.float64), np.asarray(res.directions, dtype=np.float64), ok

        _, du, oku = _trace_variant(origins, ap + eps_ap * tb_v[None, :])
        _, dv, okv = _trace_variant(origins, ap + eps_ap * tc_v[None, :])
        ok = oku & okv
        if np.any(ok):
            ddu = (du[ok] - base[ok]) / eps_ap
            ddv = (dv[ok] - base[ok]) / eps_ap
            jac = np.linalg.norm(np.cross(ddu, ddv), axis=1)
            finite = np.isfinite(jac) & (jac > 0.0)
            ok_idx = np.flatnonzero(ok)
            out_ap[ok_idx[finite]] = jac[finite].astype(np.float32, copy=False)

        oy, dy, ok_fy = _trace_variant(origins + eps_film * np.array([0.0, 1.0, 0.0])[None, :], ap)
        oz, dz, ok_fz = _trace_variant(origins + eps_film * np.array([0.0, 0.0, 1.0])[None, :], ap)
        oay, day, ok_ay = _trace_variant(origins, ap + eps_ap * tb_v[None, :])
        oaz, daz, ok_az = _trace_variant(origins, ap + eps_ap * tc_v[None, :])
        ok4 = ok_fy & ok_fz & ok_ay & ok_az
        if np.any(ok4):
            idx4 = np.flatnonzero(ok4)
            cols = [
                np.column_stack([(oy[ok4, 1] - base_o[ok4, 1]) / eps_film,
                                 (oy[ok4, 2] - base_o[ok4, 2]) / eps_film,
                                 (dy[ok4, 1] - base[ok4, 1]) / eps_film,
                                 (dy[ok4, 2] - base[ok4, 2]) / eps_film]),
                np.column_stack([(oz[ok4, 1] - base_o[ok4, 1]) / eps_film,
                                 (oz[ok4, 2] - base_o[ok4, 2]) / eps_film,
                                 (dz[ok4, 1] - base[ok4, 1]) / eps_film,
                                 (dz[ok4, 2] - base[ok4, 2]) / eps_film]),
                np.column_stack([(oay[ok4, 1] - base_o[ok4, 1]) / eps_ap,
                                 (oay[ok4, 2] - base_o[ok4, 2]) / eps_ap,
                                 (day[ok4, 1] - base[ok4, 1]) / eps_ap,
                                 (day[ok4, 2] - base[ok4, 2]) / eps_ap]),
                np.column_stack([(oaz[ok4, 1] - base_o[ok4, 1]) / eps_ap,
                                 (oaz[ok4, 2] - base_o[ok4, 2]) / eps_ap,
                                 (daz[ok4, 1] - base[ok4, 1]) / eps_ap,
                                 (daz[ok4, 2] - base[ok4, 2]) / eps_ap]),
            ]
            mats = np.stack(cols, axis=2)
            det = np.abs(np.linalg.det(mats))
            finite = np.isfinite(det) & (det > 0.0)
            out_phase[idx4[finite]] = det[finite].astype(np.float32, copy=False)
        return out_ap, out_phase

    def _append_camera_sample_records(
        self,
        *,
        batch_id: int,
        film_uv: np.ndarray,
        film_pos: np.ndarray,
        aperture_pos: np.ndarray,
        directions: np.ndarray,
        channel_idx: np.ndarray,
        shutter_weight: np.ndarray,
        sensor_weights: np.ndarray,
        film_area_pdf: float,
        aperture_area_pdf: float,
        channel_pdf: float,
        aperture_to_solid_angle_jac: Optional[np.ndarray] = None,
        solid_angle_pdf: Optional[np.ndarray] = None,
        phase_space_jac: Optional[np.ndarray] = None,
        phase_space_pdf: Optional[np.ndarray] = None,
    ) -> None:
        """Retain camera launch-domain PDFs without changing current transport."""
        n = int(np.asarray(film_uv).shape[0])
        if n <= 0:
            return
        rows = np.zeros(n, dtype=CAMERA_SAMPLE_DTYPE)
        rows["batch_id"] = np.uint32(max(0, int(batch_id)))
        rows["sample_id"] = np.arange(n, dtype=np.uint32)
        rows["film_uv"] = np.clip(np.asarray(film_uv, dtype=np.float32)[:, :2], 0.0, 1.0)
        rows["film_pos"] = np.asarray(film_pos, dtype=np.float32)[:, :3]
        rows["aperture_pos"] = np.asarray(aperture_pos, dtype=np.float32)[:, :3]
        rows["dir"] = np.asarray(directions, dtype=np.float32)[:, :3]
        rows["sensor_channel"] = np.asarray(channel_idx, dtype=np.uint8)
        nb = int(min(self.n_bands, MAX_SPECTRAL_BANDS, int(np.asarray(sensor_weights).shape[1])))
        rows["n_bands"] = np.uint16(nb)
        rows["shutter_weight"] = np.asarray(shutter_weight, dtype=np.float32)
        if nb > 0:
            rows["sensor_weight"][:, :nb] = np.asarray(sensor_weights, dtype=np.float32)[:, :nb]
        rows["film_area_pdf"] = float(film_area_pdf)
        rows["aperture_area_pdf"] = float(aperture_area_pdf)
        rows["channel_pdf"] = float(channel_pdf)
        rows["strategy_pdf"] = float(film_area_pdf) * float(aperture_area_pdf) * float(channel_pdf)
        if aperture_to_solid_angle_jac is not None:
            jac = np.asarray(aperture_to_solid_angle_jac, dtype=np.float32).reshape(-1)
            if jac.shape[0] == n:
                rows["aperture_to_solid_angle_jac"] = jac
        if solid_angle_pdf is not None:
            spdf = np.asarray(solid_angle_pdf, dtype=np.float32).reshape(-1)
            if spdf.shape[0] == n:
                rows["solid_angle_pdf"] = spdf
        if phase_space_jac is not None:
            psj = np.asarray(phase_space_jac, dtype=np.float32).reshape(-1)
            if psj.shape[0] == n:
                rows["phase_space_jac"] = psj
        if phase_space_pdf is not None:
            psp = np.asarray(phase_space_pdf, dtype=np.float32).reshape(-1)
            if psp.shape[0] == n:
                rows["phase_space_pdf"] = psp
        rows["strategy_id"] = np.uint16(CAMERA_STRATEGY_SENSOR_APERTURE)
        # The current disc launch is intentionally treated as uniform over the
        # sensor disc domain, but its jitter-and-clamp implementation is only an
        # approximation of perfect area sampling.
        rows["flags"] = CAMERA_SAMPLE_FLAG_FILM_PDF_APPROX
        rows["flags"] |= np.where(
            (rows["aperture_to_solid_angle_jac"] > 0.0) & (rows["phase_space_jac"] > 0.0),
            CAMERA_SAMPLE_FLAG_JACOBIAN_VALID,
            np.uint32(0),
        )
        self._camera_samples.append(rows)
        self.bdpt_last_camera_samples = summarize_camera_sample_rows(rows)

    def _append_optical_transfer_events(self, detailed_result, film_uv: np.ndarray, *, stream: int, batch_id: int = 0) -> None:
        """Retain per-element parametric transfer events for optics/MIS work."""
        events = tuple(getattr(detailed_result, "events", tuple()))
        if not events:
            return
        uv = np.asarray(film_uv, dtype=np.float32)
        rows = np.empty(len(events), dtype=OPTICAL_TRANSFER_DTYPE)
        for i, ev in enumerate(events):
            kind_s = str(getattr(ev, "kind", ""))
            if kind_s == "ConicSurface":
                kind = 0
            elif kind_s == "FlatSurface":
                kind = 1
            elif kind_s == "ApertureStop":
                kind = 2
            else:
                kind = 255
            ray_i = int(getattr(ev, "ray_index", 0))
            rows["batch_id"][i] = np.uint32(max(0, int(batch_id)))
            rows["ray_id"][i] = np.uint32(max(0, ray_i))
            rows["event_index"][i] = np.uint16(max(0, int(getattr(ev, "event_index", 0))))
            rows["element_idx"][i] = int(getattr(ev, "element_idx", -1))
            rows["kind"][i] = np.uint8(kind)
            reason = getattr(ev, "reason", None)
            rows["reason"][i] = np.uint8(int(getattr(reason, "value", 0)))
            rows["p0"][i] = np.asarray(getattr(ev, "p0", np.zeros(3)), dtype=np.float32)
            rows["p1"][i] = np.asarray(getattr(ev, "p1", np.zeros(3)), dtype=np.float32)
            rows["dir_in"][i] = np.asarray(getattr(ev, "dir_in", np.zeros(3)), dtype=np.float32)
            rows["dir_out"][i] = np.asarray(getattr(ev, "dir_out", np.zeros(3)), dtype=np.float32)
            rows["normal"][i] = np.asarray(getattr(ev, "normal", np.zeros(3)), dtype=np.float32)
            rows["geom_len"][i] = float(getattr(ev, "geom_len", 0.0))
            rows["opl"][i] = float(getattr(ev, "opl", 0.0))
            rows["n_before"][i] = float(getattr(ev, "n_before", 1.0))
            rows["n_after"][i] = float(getattr(ev, "n_after", 1.0))
            rows["aperture_r"][i] = float(getattr(ev, "aperture_r", 0.0))
            if uv.ndim == 2 and 0 <= ray_i < uv.shape[0] and uv.shape[1] >= 2:
                rows["film_uv"][i] = np.clip(uv[ray_i, :2], 0.0, 1.0)
            else:
                rows["film_uv"][i] = 0.0
            rows["stream"][i] = np.uint8(1 if int(stream) else 0)
            rows["flags"][i] = 0
        rows["strategy_id"]   = np.uint16(CAMERA_STRATEGY_SENSOR_APERTURE) if int(stream) else np.uint16(0)
        rows["sample_domain"] = BDPT_SAMPLE_DOMAIN_FILM_APERTURE if int(stream) else BDPT_SAMPLE_DOMAIN_HEMISPHERE
        _fill_optical_transfer_fresnel(rows)
        self._optical_transfers.append(rows)
        self.bdpt_last_optical_transfer = summarize_optical_transfer_rows(rows)

    def _async_bdpt_forward_count(self) -> int:
        return self._bdpt_endpoints.fwd_count()

    def _async_bdpt_backward_count(self) -> int:
        return self._bdpt_endpoints.bwd_count()

    def _pause_drain_loop_and_flush(self) -> None:
        """Stop the background drain loop before a shadow ray pass.

        We intentionally do NOT pre-flush the tracer output queue here.
        The _drain_shadow_pass closure already routes any non-shadow records
        that arrive during the shadow pass to _accumulate_records via the
        standard path, so a pre-flush would only duplicate that work while
        burning CPU/GPU on millions of in-flight forward/backward records.
        """
        if self._drain_thread is not None and self._drain_thread.is_alive():
            self._drain_stop.set()
            self._drain_thread.join(timeout=2.0)
        self._drain_thread = None
        self._drain_stop.clear()

    def _run_pipeline_bdpt_shadow_connections(
        self,
        records: np.ndarray,
        n: int,
        *,
        seed: int,
        max_shadow_rays: int,
        sensor_grid_res: int = 0,
    ) -> np.ndarray:
        """Connect forward/backward endpoint records using shadow rays in the live pipeline.

        ``sensor_grid_res`` is retained for older callers; current backward
        endpoints carry continuous film UV and splat into the ``n×n`` output.

        Contributions are weighted by assemble_bdpt_connection_weight() using
        provisional endpoint PDFs.  pair_scale = N_fwd/k compensates for
        sub-sampled forward pairing.  Full lens Jacobians, surface cosine
        terms, and complete path-strategy PDFs are still pending.
        """
        img = np.zeros((n, n, 3), dtype=np.float64)
        weight_img = np.zeros((n, n), dtype=np.float64)
        self.bdpt_last_shadow_stats = {
            "candidates": 0,
            "shadow_rays": 0,
            "visible": 0,
            "blocked": 0,
            "miss_records": 0,
            # Stage diagnostics — set at each filter point so the HUD can
            # report where the pipeline bottoms out.
            "n_fwd": 0,
            "n_bwd": 0,
            "n_bwd_valid": 0,
            "n_pairs": 0,
            "n_valid_dist": -1,
        }
        if records is None or records.size == 0:
            return img

        rec = np.ascontiguousarray(records, dtype=BDPT_ENDPOINT_DTYPE)
        sid = rec["stream"]
        fwd_mask = sid == 0
        bwd_mask = sid == 1
        n_fwd = int(np.count_nonzero(fwd_mask))
        n_bwd = int(np.count_nonzero(bwd_mask))
        self.bdpt_last_shadow_stats["n_fwd"] = n_fwd
        self.bdpt_last_shadow_stats["n_bwd"] = n_bwd
        if n_fwd == 0 or n_bwd == 0:
            return img

        fwd_idx_all = np.flatnonzero(fwd_mask)
        bwd_idx_all = np.flatnonzero(bwd_mask)
        bwd_uv = np.asarray(rec["film_uv"][bwd_idx_all], dtype=np.float64)
        bwd_valid = (
            np.isfinite(bwd_uv[:, 0])
            & np.isfinite(bwd_uv[:, 1])
            & (bwd_uv[:, 0] >= 0.0)
            & (bwd_uv[:, 0] <= 1.0)
            & (bwd_uv[:, 1] >= 0.0)
            & (bwd_uv[:, 1] <= 1.0)
        )
        n_bwd_valid = int(np.count_nonzero(bwd_valid))
        self.bdpt_last_shadow_stats["n_bwd_valid"] = n_bwd_valid
        if n_bwd_valid == 0:
            return img
        bwd_idx_all = bwd_idx_all[bwd_valid]
        bwd_uv = bwd_uv[bwd_valid]

        # NON-STANDARD BDPT NOTE:
        # Temporary bounded cache connector.  We sample up to k forward
        # endpoints per backward endpoint instead of enumerating the full
        # Cartesian product.  The later power scale includes N_forward/k so the
        # selected pairs estimate the full cached-endpoint sum instead of
        # becoming merely a top-k/truncated image.  This is still not full BDPT:
        # the endpoint cache lacks complete per-vertex PDFs/MIS data.
        n_bwd_act = int(bwd_idx_all.shape[0])
        n_fwd_act = int(fwd_idx_all.shape[0])
        max_pairs = int(max(1, min(max_shadow_rays, self._bdpt_shadow_batch_cap)))
        k = int(max(1, min(self._bdpt_shadow_max_fwd_per_bwd, n_fwd_act)))
        if n_bwd_act * k > max_pairs:
            take_bwd = max(1, max_pairs // max(1, k))
            start = int(self._bdpt_bwd_cursor) % max(1, n_bwd_act)
            take = (start + np.arange(take_bwd, dtype=np.int64)) % n_bwd_act
            self._bdpt_bwd_cursor = int((start + take_bwd) % max(1, n_bwd_act))
            bwd_idx_active = bwd_idx_all[take]
        else:
            bwd_idx_active = bwd_idx_all
        n_bwd_active = int(bwd_idx_active.shape[0])
        rng = np.random.default_rng(int(seed) ^ 0x5BD7)
        pair_bwd = np.repeat(bwd_idx_active, k).astype(np.int64, copy=False)
        if k >= n_fwd_act:
            pair_fwd = np.tile(fwd_idx_all, n_bwd_active).astype(np.int64, copy=False)
        else:
            sampled = rng.choice(fwd_idx_all, size=(n_bwd_active, k), replace=True)
            pair_fwd = sampled.reshape(-1).astype(np.int64, copy=False)
        pair_scale = float(n_fwd_act) / float(k)
        n_pairs_built = int(pair_bwd.shape[0])

        self.bdpt_last_shadow_stats["n_pairs"] = n_pairs_built
        if n_pairs_built == 0:
            return img
        n_pairs = int(pair_fwd.shape[0])
        if n_pairs <= 0:
            return img

        p0 = rec["pos"][pair_bwd].astype(np.float64, copy=False)
        p1 = rec["pos"][pair_fwd].astype(np.float64, copy=False)
        delta = p1 - p0
        dist = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(dist) & (dist > 1.0e-6)
        _n_valid = int(np.count_nonzero(valid))
        self.bdpt_last_shadow_stats["n_valid_dist"] = _n_valid
        # Print every invalid pair so we can see exactly which ones are bad.
        _invalid_mask = ~valid
        _n_invalid = int(np.count_nonzero(_invalid_mask))
        _inv_idx = np.where(_invalid_mask)[0]
        for _ii in _inv_idx[:200]:  # cap at 200 lines
            print(
                f"[bdpt-invalid-pair] idx={int(_ii)}"
                f" dist={float(dist[_ii]):.6e}"
                f" p0={p0[_ii].tolist()}"
                f" p1={p1[_ii].tolist()}"
                f" pair_bwd={int(pair_bwd[_ii])} pair_fwd={int(pair_fwd[_ii])}"
                f" rec_p0={rec['pos'][pair_bwd[_ii]].tolist()}"
                f" rec_p1={rec['pos'][pair_fwd[_ii]].tolist()}",
                flush=True,
            )
        print(
            f"[bdpt-dist] n_pairs={int(dist.shape[0])} n_valid={_n_valid} n_invalid={_n_invalid}"
            f" dist_min={float(np.nanmin(dist)):.6e} dist_max={float(np.nanmax(dist)):.6e}"
            f" dist[:5]={dist[:min(5,dist.shape[0])].tolist()}"
            f" p0[:2]={p0[:min(2,p0.shape[0])].tolist()}"
            f" p1[:2]={p1[:min(2,p1.shape[0])].tolist()}",
            flush=True,
        )
        if not np.any(valid):
            # Always print so we can see what the actual positions are.
            _nb = min(5, int(p0.shape[0]))
            _nan_p0  = int(np.count_nonzero(~np.isfinite(p0)))
            _nan_p1  = int(np.count_nonzero(~np.isfinite(p1)))
            _zero_p0 = int(np.count_nonzero(np.all(p0 == 0.0, axis=1)))
            _zero_p1 = int(np.count_nonzero(np.all(p1 == 0.0, axis=1)))
            _inf_d   = int(np.count_nonzero(~np.isfinite(dist)))
            _tiny_d  = int(np.count_nonzero(np.isfinite(dist) & (dist <= 1e-6)))
            print(
                f"[BDPT-INVALID-DIST]"
                f" n_pairs={int(dist.shape[0])}"
                f" n_fwd_recs={n_fwd} n_bwd_recs={n_bwd}"
                f" nan_p0={_nan_p0} nan_p1={_nan_p1}"
                f" zero_p0={_zero_p0} zero_p1={_zero_p1}"
                f" inf_dist={_inf_d} tiny_dist(<=1e-6)={_tiny_d}"
                f"\n  dist[:5]={dist[:_nb]}"
                f"\n  p0[:5]={p0[:_nb]}"
                f"\n  p1[:5]={p1[:_nb]}"
                f"\n  fwd_pos_range x=[{np.nanmin(p1[:,0]):.4f},{np.nanmax(p1[:,0]):.4f}]"
                f" y=[{np.nanmin(p1[:,1]):.4f},{np.nanmax(p1[:,1]):.4f}]"
                f" z=[{np.nanmin(p1[:,2]):.4f},{np.nanmax(p1[:,2]):.4f}]"
                f"\n  bwd_pos_range x=[{np.nanmin(p0[:,0]):.4f},{np.nanmax(p0[:,0]):.4f}]"
                f" y=[{np.nanmin(p0[:,1]):.4f},{np.nanmax(p0[:,1]):.4f}]"
                f" z=[{np.nanmin(p0[:,2]):.4f},{np.nanmax(p0[:,2]):.4f}]",
                flush=True,
            )
            return img
        pair_fwd = pair_fwd[valid]
        pair_bwd = pair_bwd[valid]
        p0 = p0[valid]
        delta = delta[valid]
        dist = dist[valid]
        dirs = delta / np.maximum(dist[:, None], 1.0e-30)
        n_pairs = int(pair_fwd.shape[0])

        # ── Shared drain-loop helper ─────────────────────────────────────────────
        # Drains records from the tracer, updating first_hit[local_idx] for any
        # records whose color_flag matches shadow_cflag, and re-routing everything
        # else into _accumulate_records.  Returns (first_hit, miss_count).
        _SHADOW_TAG_HI = np.uint64(0xBD)

        def _drain_shadow_pass(
            n_pts: int,
            shadow_cflag: int,
            ray_origins: np.ndarray,
            _first_hit: np.ndarray,
        ) -> tuple[np.ndarray, int]:
            _mc = 0
            # Each submitted shadow ray produces exactly one "primary" record:
            # STRIKE (kind=0) for a hit, or MISS (kind=2) for no hit.  Count them
            # so we can break as soon as every shadow ray has been accounted for,
            # regardless of what else is in-flight in the main pipeline.
            # NOTE: GPU-mode MISS rays (no intersection) do NOT emit a MISS record;
            # they just decrement in_flight silently.  The timeout below catches
            # these cases — any unaccounted pair keeps first_hit=inf, which maps
            # to visible=True (correct: nothing blocked the path).
            _shadow_primary_seen = 0
            _t_start = time.perf_counter()
            _last_progress_t = _t_start
            _last_progress_seen = 0
            _last_in_flight = self.tracer.in_flight_count()

            def _process_batch(rec: dict, nr: int) -> None:
                nonlocal _mc, _shadow_primary_seen
                _rt_all = np.asarray(rec["tag"], dtype=np.uint64)
                # Identify shadow rays by the 0xBD high-byte tag — NOT by cflag,
                # because the tracer overwrites cflag with the surface material's value.
                _sh = (_rt_all >> np.uint64(56)) == _SHADOW_TAG_HI
                if np.any(_sh):
                    _rt = _rt_all[_sh]
                    _lc = (_rt & np.uint64(0x0000FFFFFFFFFFFF)).astype(np.int64)
                    _ir = (_lc >= 0) & (_lc < n_pts)
                    _lc = _lc[_ir]
                    if _lc.size > 0:
                        _kd = np.asarray(rec["kind"])[_sh][_ir]
                        _is_strike = _kd == 0   # primary hit record
                        _is_miss   = _kd == 2   # primary miss record
                        # TERMINAL (kind=1) records are secondary; don't count them.
                        _shadow_primary_seen += int(np.count_nonzero(_is_strike | _is_miss))
                        _mc += int(np.count_nonzero(_is_miss))
                        if np.any(_is_strike):
                            _hp = np.asarray(rec["pos"], dtype=np.float32)[_sh][_ir][_is_strike].astype(np.float64)
                            _hd = np.linalg.norm(_hp - ray_origins[_lc[_is_strike]], axis=1)
                            np.minimum.at(_first_hit, _lc[_is_strike], _hd)
                if np.any(~_sh):
                    _kp = {_k: np.asarray(_v)[~_sh] for _k, _v in rec.items()
                           if hasattr(_v, "__len__") and len(_v) == nr}
                    if _kp:
                        self._fast_bdpt_feed(_kp)
                        self._accumulate_records(_kp)

            while True:
                _r = self.tracer.drain_records_slim(max_n=BDPT_PIPELINE_CAP)
                _nr = int(_r.get("kind", np.array([], dtype=np.uint8)).shape[0]) if _r else 0
                if _nr > 0:
                    _process_batch(_r, _nr)
                # Primary termination: all n_pts shadow rays have had their first
                # interaction (STRIKE or MISS).  The main pipeline may still have
                # other rays in-flight — we do NOT wait for those.
                if _shadow_primary_seen >= n_pts:
                    break
                # Fallback: if the entire pipeline goes idle before we see all
                # expected primary records (e.g. due to dropped rays), drain
                # whatever remains and stop rather than spinning forever.
                if self.tracer.in_flight_count() == 0:
                    _tl = self.tracer.drain_records_slim(max_n=BDPT_PIPELINE_CAP)
                    _nt = int(_tl.get("kind", np.array([], dtype=np.uint8)).shape[0]) if _tl else 0
                    if _nt > 0:
                        _process_batch(_tl, _nt)
                    break
                # Progress-stall timeout: reset timer whenever _shadow_primary_seen
                # advances OR in_flight_count() drops.  GPU-mode miss rays don't
                # emit MISS records, so seen can plateau while in_flight still ticks
                # down — without the in_flight check the stall fires after 5 s even
                # when the GPU is actively processing all shadow rays.
                _now = time.perf_counter()
                _cur_in_flight = self.tracer.in_flight_count()
                if _shadow_primary_seen > _last_progress_seen or _cur_in_flight < _last_in_flight:
                    _last_progress_t    = _now
                    _last_progress_seen = _shadow_primary_seen
                    _last_in_flight     = _cur_in_flight
                _stall_s = _now - _last_progress_t
                _total_s = _now - _t_start
                if _stall_s > 30.0 or _total_s > 120.0:
                    print(
                        f"[bdpt-shadow-timeout] seen={_shadow_primary_seen}/{n_pts}"
                        f" stall={_stall_s:.1f}s total={_total_s:.1f}s"
                        f" in_flight={self.tracer.in_flight_count()}",
                        flush=True,
                    )
                    _tl = self.tracer.drain_records_slim(max_n=BDPT_PIPELINE_CAP)
                    _nt = int(_tl.get("kind", np.array([], dtype=np.uint8)).shape[0]) if _tl else 0
                    if _nt > 0:
                        _process_batch(_tl, _nt)
                    break
                time.sleep(0.001)
            return _first_hit, _mc

        # ── Pre-compute wavelength→RGB weights (shared by both passes) ───────
        wl_nm = (C_LIGHT / np.maximum(
            np.asarray(self.freq_hz[:self.n_bands], dtype=np.float64), EPS)) * 1.0e9
        rgb_w = _wavelength_to_rgb_weights(wl_nm).astype(np.float64)
        # Sensor spectral reactance: sens_rgb_w[ch, band] = sensor's response to
        # that frequency band for RGB channel ch.  Used as the coupling weight
        # when pairing forward (any band) with backward endpoints.
        sens_rgb_w = _sensor_rgb_sensitivity_bands(
            self.freq_hz[:self.n_bands]).astype(np.float64)  # (3, n_bands)
        flat = img.reshape(-1, 3).astype(np.float64, copy=False)

        # ── Pass 1: fwd↔bwd endpoint pairs ──────────────────────────────────
        eps = 1.0e-5
        origins = p0 + dirs * eps
        target_dist = np.maximum(dist - 2.0 * eps, 0.0)
        tags = (np.uint64(self._bdpt_shadow_tag_base)
                | np.arange(n_pairs, dtype=np.uint64))
        src_ids = np.arange(n_pairs, dtype=np.int32)
        amps = np.ones((n_pairs, int(self.n_bands)), dtype=np.complex128)
        cflags = np.full(n_pairs, 2, dtype=np.uint8)
        first_hit = np.full(n_pairs, np.inf, dtype=np.float64)
        miss_count = 0

        self._pause_drain_loop_and_flush()
        try:
            self.tracer.submit_rays(
                origins=np.ascontiguousarray(origins),
                directions=np.ascontiguousarray(dirs),
                amplitudes=np.ascontiguousarray(amps),
                src_ids=np.ascontiguousarray(src_ids),
                tags=np.ascontiguousarray(tags),
                color_flags=np.ascontiguousarray(cflags),
                max_bounces=0,
                min_amplitude=0.0,
                max_children=0,
                seed=int(seed ^ 0x51A0),
                use_gpu_compute=(self.compute_mode in ("gpu", "mixed")),
                gpu_all_stages=(self.compute_mode == "gpu"),
                shader_dir=_SHADER_DIR,
            )
            first_hit, miss_count = _drain_shadow_pass(n_pairs, 2, origins, first_hit)
        finally:
            self._ensure_drain_loop()

        visible = first_hit >= (target_dist - 5.0e-5)
        _n_vis = int(np.count_nonzero(visible))
        print(
            f"[bdpt-shadow1] n_pairs={int(n_pairs)} miss_count={miss_count}"
            f" visible={_n_vis} blocked={int(n_pairs)-_n_vis}"
            f" first_hit[:5]={first_hit[:5].tolist()}"
            f" target_dist[:5]={target_dist[:5].tolist()}",
            flush=True,
        )
        if np.any(visible):
            vf = pair_fwd[visible]
            vb = pair_bwd[visible]
            # Weight by the sensor's spectral reactance at the forward ray's
            # band.  The backward amplitude encodes sensor importance; the
            # forward band selects the sensor response curve.
            bands_fwd = rec["band"][vf].astype(np.int64)
            bands_clamped = np.clip(bands_fwd, 0, self.n_bands - 1)
            fa = rec["amp"][vf].astype(np.complex128)
            ba = rec["amp"][vb].astype(np.complex128)
            power = assemble_bdpt_connection_weight(
                fa, ba, dist[visible],
                fwd_pdf=rec["launch_pdf"][vf].astype(np.float64),
                bwd_pdf=rec["launch_pdf"][vb].astype(np.float64),
                pair_scale=pair_scale,
            )
            print(
                f"[bdpt-pass1] visible={_n_vis}"
                f" power_min={float(np.min(power)):.3e}"
                f" power_max={float(np.max(power)):.3e}"
                f" power_sum={float(np.sum(power)):.3e}"
                f" p_fwd={float(np.mean(rec['launch_pdf'][vf])):.3e}"
                f" p_bwd={float(np.mean(rec['launch_pdf'][vb])):.3e}"
                f" fa[:3]={np.abs(fa[:3]).tolist()}"
                f" ba[:3]={np.abs(ba[:3]).tolist()}",
                flush=True,
            )
            uv = np.clip(rec["film_uv"][vb].astype(np.float64), 0.0, 1.0 - 1.0e-7)
            fx = uv[:, 0] * float(max(1, n - 1))
            fy = uv[:, 1] * float(max(1, n - 1))
            x0 = np.floor(fx).astype(np.int64)
            y0 = np.floor(fy).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, n - 1)
            y1 = np.clip(y0 + 1, 0, n - 1)
            tx = fx - x0.astype(np.float64)
            ty = fy - y0.astype(np.float64)
            idx00 = y0 * n + x0
            idx10 = y0 * n + x1
            idx01 = y1 * n + x0
            idx11 = y1 * n + x1
            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = tx * (1.0 - ty)
            w01 = (1.0 - tx) * ty
            w11 = tx * ty
            for ch in range(3):
                val = power * sens_rgb_w[ch, bands_clamped]
                np.add.at(flat[:, ch], idx00, val * w00)
                np.add.at(flat[:, ch], idx10, val * w10)
                np.add.at(flat[:, ch], idx01, val * w01)
                np.add.at(flat[:, ch], idx11, val * w11)
            weight_flat = weight_img.reshape(-1)
            np.add.at(weight_flat, idx00, w00)
            np.add.at(weight_flat, idx10, w10)
            np.add.at(weight_flat, idx01, w01)
            np.add.at(weight_flat, idx11, w11)
            print(
                f"[bdpt-accum] flat_max={float(np.max(flat)):.3e}"
                f" flat_nonzero={int(np.count_nonzero(flat))}"
                f" splat_samples={int(uv.shape[0])}"
                f" uv_range=[({float(np.min(uv[:,0])):.4f},{float(np.max(uv[:,0])):.4f}),"
                f"({float(np.min(uv[:,1])):.4f},{float(np.max(uv[:,1])):.4f})]"
                f" sens_w_max={float(np.max(sens_rgb_w)):.3e}",
                flush=True,
            )

        self.bdpt_last_shadow_stats.update({
            "candidates": int(n_pairs),
            "shadow_rays": int(n_pairs),
            "visible": _n_vis,
            "blocked": int(n_pairs) - _n_vis,
            "miss_records": int(miss_count),
            "direct_vis": 0,
            "direct_total": 0,
        })

        _out_max = float(np.max(img))
        _out_nonzero = int(np.count_nonzero(img))
        print(
            f"[bdpt-return] img_max={_out_max:.3e} img_nonzero={_out_nonzero}"
            f" weight_nonzero={int(np.count_nonzero(weight_img))}"
            f" img_dtype={img.dtype} img_shape={img.shape}",
            flush=True,
        )
        return img.astype(np.float32, copy=False), weight_img.astype(np.float32, copy=False)

    def _report_aperture_aim_extrema(self) -> dict:
        """Compute and print the valid backward-ray cone for representative sensor sites.

        For a pixel at transverse height h from the optical axis, and an aperture
        disk at axial distance d with clear radius r_ap, the valid backward ray
        directions are those that intersect the disk.  In the meridional plane the
        cone spans [theta_near, theta_far]:

            theta_near = atan2(h - r_ap, d)   (near aperture rim; can be negative)
            theta_far  = atan2(h + r_ap, d)   (far aperture rim)
            theta_chief = atan2(h, d)          (aimed at aperture centre)

        The solid-angle fraction relative to a hemisphere is stored as
        ``accept_fraction``.  A fraction near zero means almost no backward rays
        launched toward the hemisphere will reach the aperture.

        Results are stored in self._aperture_aim_extrema and printed once.
        """
        ap_cen = getattr(self, "aperture_centroid", None)
        ap_r   = float(getattr(self, "aperture_radius", 0.0))
        plate  = self.scene.image_plate
        plate_x = float(plate.x)
        plate_r = float(plate.radius)

        if ap_cen is None or ap_r <= 0.0:
            print("[aim-extrema] aperture not yet set", flush=True)
            self._aperture_aim_extrema = {}
            return {}

        ap_x = float(np.asarray(ap_cen, dtype=np.float64)[0])
        d = abs(plate_x - ap_x)
        if d < 1.0e-6:
            print("[aim-extrema] aperture and sensor are coplanar", flush=True)
            self._aperture_aim_extrema = {}
            return {}

        # Check camera-frustum constraint: backward rays must also pass through
        # the frustum aperture at scene.exit_pupil_x / scene.exit_pupil_radius.
        frust_x = float(getattr(self.scene, "exit_pupil_x", 0.0))
        frust_r = float(getattr(self.scene, "exit_pupil_radius", 0.0))
        has_frust = frust_r > 0.0 and 0.0 < frust_x < plate_x

        results = {}
        labels_and_heights = [
            ("center", 0.0),
            ("mid",    0.5 * plate_r),
            ("edge",   plate_r),
        ]
        for label, h in labels_and_heights:
            theta_near  = math.atan2(h - ap_r, d)
            theta_far   = math.atan2(h + ap_r, d)
            theta_chief = math.atan2(h, d)
            cone_half   = math.atan2(ap_r, math.sqrt(d * d + h * h))
            # Solid angle of aperture disk from this pixel (paraxial approx).
            dist2 = d * d + h * h
            solid_angle = math.pi * ap_r * ap_r / dist2
            accept_frac = solid_angle / (2.0 * math.pi)   # fraction of hemisphere

            # Frustum constraint: which portion of the aperture disk is reachable
            # without being blocked by the camera-body frustum cone?
            frust_note = ""
            if has_frust and frust_x > ap_x:
                df = abs(plate_x - frust_x)          # sensor → frustum plane
                # Max aperture y-coordinate visible from pixel h=h through frustum r=frust_r:
                # y_ap_max satisfies: h + (y_ap_max - h)*df/d = frust_r
                # → y_ap_max = h + (frust_r - h) * d / df
                y_ap_max = h + (frust_r - h) * d / df
                r_visible = max(0.0, min(ap_r, y_ap_max - 0.0))
                # Near side: y_ap_min from the negative frustum boundary
                y_ap_min_neg = h + (-frust_r - h) * d / df
                r_blocked_near = max(0.0, -(y_ap_min_neg))   # how much of -y side is blocked
                visible_frac = max(0.0, min(1.0, (r_visible + r_blocked_near) / (2.0 * ap_r)))
                accept_frac *= visible_frac
                frust_note = f" frust_visible_frac={visible_frac:.2f}"

            entry = {
                "h_sensor_m": h,
                "ap_x_m": ap_x,
                "ap_r_m": ap_r,
                "d_m": d,
                "theta_near_rad":  theta_near,
                "theta_far_rad":   theta_far,
                "theta_chief_rad": theta_chief,
                "cone_half_angle_rad": cone_half,
                "solid_angle_sr":  solid_angle,
                "accept_fraction": accept_frac,
            }
            results[label] = entry
            print(
                f"[aim-extrema:{label}]"
                f"  h={h*1e3:.1f}mm"
                f"  ap_x={ap_x:.3f}  r={ap_r*1e3:.1f}mm  d={d:.3f}"
                f"  near={math.degrees(theta_near):.1f}°"
                f"  chief={math.degrees(theta_chief):.1f}°"
                f"  far={math.degrees(theta_far):.1f}°"
                f"  half={math.degrees(cone_half):.2f}°"
                f"  accept={accept_frac*100:.3f}%{frust_note}",
                flush=True,
            )

        self._aperture_aim_extrema = results
        return results

    def get_ray_segments(self) -> Optional[np.ndarray]:
        """Superseded by the voxel display grid (_disp_grid). Returns None."""
        return None

    def get_bdpt_records(self) -> Optional[np.ndarray]:
        """Return last captured BDPT endpoint records in legacy debug layout."""
        with self._trace_lock:
            return _bdpt_compact_to_legacy_records(self._last_bdpt_records)

    def get_bdpt_segments(self) -> Optional[np.ndarray]:
        """Return retained BDPT subsegment records for future MIS estimators."""
        with self._trace_lock:
            return self._bdpt_segments.snapshot()

    def get_optical_transfers(self) -> Optional[np.ndarray]:
        """Return retained parametric lens transfer event records."""
        with self._trace_lock:
            return self._optical_transfers.snapshot()

    def get_camera_samples(self) -> Optional[np.ndarray]:
        """Return retained backward camera launch sample/PDF records."""
        with self._trace_lock:
            return self._camera_samples.snapshot()

    def get_forward_strike_image(self) -> np.ndarray:
        """Return the pure forward-traced image-plate accumulation preview."""
        with self._segs_lock:
            img = self._forward_img_accum.copy()
        flat = np.asarray(img, dtype=np.float64).ravel()
        pos = flat[flat > 0.0]
        if pos.size == 0:
            return np.zeros(img.shape, dtype=np.float32)
        # Auto-normalise to 99th-percentile so the projection becomes visible
        # as soon as any hits arrive, regardless of absolute amplitude scale.
        white = float(np.percentile(pos, 99.0))
        white = max(white, 1.0e-30)
        y = np.log1p(np.maximum(img, 0.0) / white * 6.0) / np.log1p(6.0)
        return np.ascontiguousarray(np.clip(y, 0.0, 1.0), dtype=np.float32)

    def get_reverse_strike_image(self) -> np.ndarray:
        """Return the pure reverse-traced strike distribution preview."""
        with self._segs_lock:
            img = self._reverse_img_accum.copy()
        disp = np.log1p(np.maximum(img, 0.0) * float(self._reverse_img_gain))
        return np.ascontiguousarray(disp / (1.0 + disp), dtype=np.float32)

    def build_all_prospective_reverse_segments(
        self,
        seed: int,
        max_segments: int = 25_000_000,
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
            # Prefer the optical exit pupil (assembly-computed) over the physical
            # body position stored in scene.exit_pupil_x, which may be the G4
            # back face rather than where rays should actually be aimed.
            if self._lens_assembly is not None:
                _bdpt_cen, _bdpt_r = self._lens_assembly.backward_ray_target()
                if _bdpt_cen is not None and _bdpt_r > 0.0:
                    stop_plane_x = float(_bdpt_cen[0])
                    stop_radius_m = float(_bdpt_r)
            else:
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
                + (px_grid[:, :, None] + 0.5 - 0.5 * float(n_px)) * pix_w * right[None, None, :]
                + (py_grid[:, :, None] + 0.5 - 0.5 * float(n_py)) * pix_h * up[None, None, :]
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

    def trace_forward_backward_sensor_rgb(
        self,
        pixels: int | None = None,
        aperture_samples: int = 2,
        seed: int = 0,
        max_records: int = 25_000_000,
        max_bounces: int | None = None,
        leak: float = 1.0,
        n_rays_bdpt: int | None = None,
        camera_mode: int = 2,
    ) -> np.ndarray:
        """Render the BDPT plate from live async endpoints plus pipeline shadow rays.

        Returns
        -------
        np.ndarray
            Shape ``(n, n, 3)``, dtype ``float32``, values in ``[0, 1]``.
        """
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
            _ = (max_bounces, leak, n_rays_bdpt)
            self.bdpt_last_launched_rays = 0
            async_records = self._bdpt_endpoints.snapshot()
            # Print endpoint store state + sample positions on every camera render.
            _snap_none = async_records is None
            _snap_sz   = 0 if _snap_none else int(async_records.shape[0])
            _snap_fwd  = 0 if _snap_none else int(np.count_nonzero(async_records["stream"] == 0))
            _snap_bwd  = 0 if _snap_none else int(np.count_nonzero(async_records["stream"] == 1))
            if not _snap_none and _snap_sz > 0:
                _fwd_rows = async_records[async_records["stream"] == 0]
                _bwd_rows = async_records[async_records["stream"] == 1]
                _fwd_pos3 = _fwd_rows["pos"][:3]
                _bwd_pos3 = _bwd_rows["pos"][:3]
                _zero_fwd = int(np.count_nonzero(np.all(_fwd_rows["pos"] == 0.0, axis=1)))
                _zero_bwd = int(np.count_nonzero(np.all(_bwd_rows["pos"] == 0.0, axis=1)))
                _nan_fwd  = int(np.count_nonzero(~np.isfinite(_fwd_rows["pos"])))
                _nan_bwd  = int(np.count_nonzero(~np.isfinite(_bwd_rows["pos"])))
                print(
                    f"[bdpt-snap] rows={_snap_sz} fwd={_snap_fwd} bwd={_snap_bwd}"
                    f" | fwd_zero_pos={_zero_fwd} fwd_nan_pos={_nan_fwd}"
                    f" | bwd_zero_pos={_zero_bwd} bwd_nan_pos={_nan_bwd}"
                    f"\n  fwd_pos_sample={_fwd_pos3.tolist()}"
                    f"\n  bwd_pos_sample={_bwd_pos3.tolist()}",
                    flush=True,
                )
            else:
                print(f"[bdpt-snap] is_none={_snap_none} rows={_snap_sz}", flush=True)
            if async_records is None or async_records.size == 0:
                records = np.zeros(0, dtype=BDPT_ENDPOINT_DTYPE)
                self._last_bdpt_records = None
            else:
                records = np.array(async_records, dtype=BDPT_ENDPOINT_DTYPE, copy=True)
                self._last_bdpt_records = records
            self.bdpt_last_records = int(records.shape[0])
            self.bdpt_last_volume_records = 0
            self.bdpt_last_volume_power = 0.0

            # ── Stream-type validation counts ──────────────────────────────
            # Classify endpoint records by stream origin.  Primary: read the
            # stream_id field (ENDPOINT_DTYPE col-15, float32) written by the
            # C++ kernel — 0.0 = BDPT_SIDE_LIGHT (forward), 1.0 =
            # BDPT_SIDE_SENSOR (backward/PIXEL_CONE).  Fallback: use the
            # vertex_index sign heuristic (col-3 int32) for records produced
            # by older builds that still have the _pad field.
            #   forward light:   stream_id == 0.0  (or vertex_index < 0)
            #   backward sensor: stream_id == 1.0  (or vertex_index >= 0)
            _stream_fwd_count  = 0
            _stream_bwd_count  = 0
            _stream_pixcone_count = 0
            _fwd_mask = None
            _bwd_mask = None
            if self._last_bdpt_records is not None and self._last_bdpt_records.shape[0] > 0:
                _r = self._last_bdpt_records
                _sids  = _r["id"]
                _sid = _r["stream"]
                _fwd_mask = _sid == 0
                _bwd_mask = _sid == 1
                _stream_fwd_count  = int(np.count_nonzero(_fwd_mask))
                _stream_bwd_count  = int(np.count_nonzero(_bwd_mask))
                _pixel_cap = int(n) * int(n)
                _stream_pixcone_count = int(np.count_nonzero(_sids < np.uint32(_pixel_cap)))
            self.bdpt_last_stream_counts = {
                "forward_light":     _stream_fwd_count,
                "backward_sensor":   _stream_bwd_count,
                "pixel_cone":        _stream_pixcone_count,
                "field_deposit_blocked": _stream_bwd_count,
            }

            _sensor_grid_res = int(max(4, self.scene.image_plate.sensor_res))
            _shadow_n            = n
            _shadow_seed         = int(seed)
            _shadow_max_rays     = int(max(1, max_records))
            _shadow_sgr          = _sensor_grid_res
            # Snapshot is taken; release _trace_lock before running the shadow
            # connection pipeline.  Shadow passes pause the drain loop internally
            # and can take seconds — holding the lock for that entire time would
            # starve the main _trace thread of its forward/backward submissions.

        # ── Shadow connection pipeline — runs WITHOUT _trace_lock ────────────
        out_result = self._run_pipeline_bdpt_shadow_connections(
            records,
            _shadow_n,
            seed=_shadow_seed,
            max_shadow_rays=_shadow_max_rays,
            sensor_grid_res=_shadow_sgr,
        )
        if isinstance(out_result, tuple):
            out, out_weight = out_result
        else:
            out = out_result
            out_weight = np.zeros((_shadow_n, _shadow_n), dtype=np.float32)

        self.bdpt_last_survivor_records = int(self.bdpt_last_shadow_stats.get("visible", 0))
        if self.bdpt_last_survivor_records <= 0:
            self.bdpt_consecutive_no_survivor_frames += 1
        else:
            self.bdpt_consecutive_no_survivor_frames = 0
        out_lin = np.asarray(out, dtype=np.float64)
        if self._bdpt_plate_linear_accum is None or self._bdpt_plate_linear_accum.shape != out_lin.shape:
            self._bdpt_plate_linear_accum = np.zeros_like(out_lin, dtype=np.float64)
            self._bdpt_plate_weight_accum = np.zeros(out_lin.shape[:2], dtype=np.float64)
            self._bdpt_plate_sample_count = 0
            self._bdpt_bwd_cursor = 0
        if int(np.count_nonzero(out_lin)) > 0:
            self._bdpt_plate_linear_accum += out_lin
            if self._bdpt_plate_weight_accum is not None:
                self._bdpt_plate_weight_accum += np.asarray(out_weight, dtype=np.float64)
            self._bdpt_plate_sample_count += int(self.bdpt_last_shadow_stats.get("shadow_rays", 0))

        accum_lin = self._bdpt_plate_linear_accum
        accum_w = self._bdpt_plate_weight_accum
        self.bdpt_last_sensor_photons = float(np.sum(np.asarray(accum_lin, dtype=np.float64)))
        self.bdpt_last_sensor_power = self.bdpt_last_sensor_photons
        self.bdpt_last_telemetry = {
            "kept_records": self.bdpt_last_survivor_records,
            "shadow_candidates": int(self.bdpt_last_shadow_stats.get("candidates", 0)),
            "shadow_rays": int(self.bdpt_last_shadow_stats.get("shadow_rays", 0)),
            "shadow_blocked": int(self.bdpt_last_shadow_stats.get("blocked", 0)),
            "optical_events": int(self.bdpt_last_optical_transfer.get("events", 0)),
            "optical_failed_rays": int(self.bdpt_last_optical_transfer.get("failed_rays", 0)),
            "camera_samples": int(self.bdpt_last_camera_samples.get("samples", 0)),
        }

        lit_mask = np.sum(np.asarray(accum_lin, dtype=np.float64), axis=2) > 1.0e-8
        lit_pixels = int(np.count_nonzero(lit_mask))
        lit_fraction = float(lit_pixels) / float(max(1, _shadow_n * _shadow_n))
        self.bdpt_debug_print_counter += 1
        print(
            "[bdpt-debug]",
            f"call={self.bdpt_debug_print_counter}",
            f"launched={self.bdpt_last_launched_rays}",
            f"endpoint_records={self.bdpt_last_records}",
            f"fwd={_stream_fwd_count}",
            f"bwd={_stream_bwd_count}(pixcone={_stream_pixcone_count})",
            f"field_blocked={self.bdpt_last_stream_counts['field_deposit_blocked']}",
            f"shadow={int(self.bdpt_last_shadow_stats.get('shadow_rays', 0))}",
            f"visible={self.bdpt_last_survivor_records}",
            f"blocked={int(self.bdpt_last_shadow_stats.get('blocked', 0))}",
            f"lit_pixels={lit_pixels}",
            f"lit_frac={lit_fraction:.4f}",
            f"sensor_gid={sensor_gid}",
            f"bwd_emit={self.bdpt_last_backward_transport.get('found_emission', 0)}",
            f"bwd_noemit={self.bdpt_last_backward_transport.get('terminated_no_emission', 0)}",
            f"bwd_miss={self.bdpt_last_backward_transport.get('missed_scene', 0)}",
            f"cam_samples={int(self.bdpt_last_camera_samples.get('samples', 0))}",
            f"cam_pdf={float(self.bdpt_last_camera_samples.get('strategy_pdf_mean', 0.0)):.3e}",
            f"opt_events={int(self.bdpt_last_optical_transfer.get('events', 0))}",
            f"opt_fail={self.bdpt_last_optical_transfer.get('top_fail_element', -1)}:{self.bdpt_last_optical_transfer.get('top_fail_reason', '')}",
            f"photons={self.bdpt_last_sensor_photons:.3e}",
            f"sensor_power={self.bdpt_last_sensor_power:.3e}",
            f"integrated_shadow_samples={self._bdpt_plate_sample_count}",
            flush=True,
        )
        # Display is tone-mapped from the persistent linear BDPT image.  The
        # linear buffer is never replaced by a single frame, so every accepted
        # connection sample remains in the image history.
        disp = np.zeros_like(accum_lin, dtype=np.float64)
        if accum_w is not None:
            display_lin = accum_lin / np.maximum(accum_w[:, :, None], 1.0e-12)
        else:
            display_lin = accum_lin
        pos_vals = display_lin[display_lin > 0.0]
        if pos_vals.size > 0:
            white = float(np.percentile(pos_vals, 99.0))
            disp[:] = np.log1p(np.maximum(display_lin, 0.0) / max(white, 1.0e-30) * 6.0) / np.log1p(6.0)
        self._last_bdpt_plate_rgb = np.ascontiguousarray(np.clip(disp, 0.0, 1.0), dtype=np.float32)
        return self._last_bdpt_plate_rgb

    def _do_register_neural_assembly_group(self) -> None:
        """Re-register neural assembly after clear_tri_groups().  No-op if no MLP loaded."""
        if self._lens_assembly is None or self._lens_assembly.mode != LensAssemblySpec.MODE_MLP:
            return
        lsg = getattr(self, "lens_surface_groups", None)
        if not lsg:
            return
        self._lens_assembly.register(
            self.tracer,
            lsg,
            self.tri_vertices,
            self.tri_centroids,
            _scene_lenses(self.scene),
        )

    def _do_register_parametric_assembly(self) -> None:
        """Re-register parametric lens assembly after clear_tri_groups().  No-op if not PARAMETRIC."""
        if self._lens_assembly is None or self._lens_assembly.mode != LensAssemblySpec.MODE_PARAMETRIC:
            return
        lsg = getattr(self, "lens_surface_groups", None)
        if not lsg:
            return
        self._lens_assembly.register(
            self.tracer,
            lsg,
            self.tri_vertices,
            self.tri_centroids,
            _scene_lenses(self.scene),
        )
        self._start_progressive_refinement()

    def _do_register_lut_assembly(self) -> None:
        """Re-register LUT lens assembly after clear_tri_groups() or after baking.
        No-op if assembly is not in LUT mode or grid not yet built."""
        if self._lens_assembly is None or self._lens_assembly.mode != LensAssemblySpec.MODE_LUT:
            return
        if self._lens_assembly._transfer_grid is None:
            return
        lsg = getattr(self, "lens_surface_groups", None)
        if not lsg:
            return
        self._lens_assembly.register(
            self.tracer,
            lsg,
            self.tri_vertices,
            self.tri_centroids,
            _scene_lenses(self.scene),
        )

    def _adjust_lens_element(self, idx: int, delta_x: float) -> None:
        """Shift lens element `idx` along X by `delta_x` metres and recast all ray correlation.

        Works for any element — focus group, floating compensator, variator, or front group.
        Bypasses apply_to_scene() so the solver never overrides the live adjustment.
        """
        import dataclasses as _dc
        lenses = list(getattr(self.scene, "lens_stack", None) or [])
        if not lenses:
            return
        idx = int(idx) % len(lenses)
        lenses[idx] = _dc.replace(lenses[idx], center_x=round(lenses[idx].center_x + delta_x, 6))
        self.scene.lens_stack = lenses

        if self._lens_assembly is not None and self._lens_assembly.mode == LensAssemblySpec.MODE_PARAMETRIC:
            new_cl = _compound_lens_from_stack(lenses, getattr(self.scene, "iris_aperture", None))
            self._lens_assembly.set_optics(new_cl)
            lsg = getattr(self, "lens_surface_groups", None)
            if lsg:
                saved_design = getattr(self.scene, "optical_design", None)
                self.scene.optical_design = None
                try:
                    self.tracer.clear_tri_groups()
                    self._lens_assembly.register(
                        self.tracer,
                        lsg,
                        self.tri_vertices,
                        self.tri_centroids,
                        _scene_lenses(self.scene),
                    )
                finally:
                    self.scene.optical_design = saved_design
                self._start_progressive_refinement()

        # Clear display accumulators on lens move so the image refreshes.
        # Do NOT clear _bdpt_endpoints: it holds physically-valid hit positions
        # that remain usable by the shadow connection pass across lens nudges.
        self._forward_img_accum[:] = 0.0
        self._reverse_img_accum[:] = 0.0
        if isinstance(self._backward_transport_accum, dict):
            for v in self._backward_transport_accum.values():
                if isinstance(v, np.ndarray):
                    v[:] = 0.0
        self._last_bdpt_plate_rgb = None
        self._last_bdpt_records = None
        self._bdpt_plate_linear_accum = None
        self._bdpt_plate_weight_accum = None
        self._bdpt_plate_sample_count = 0
        self._bdpt_bwd_cursor = 0
        self._bdpt_segments.clear()
        self._optical_transfers.clear()
        self._camera_samples.clear()
        self._backward_launch_pdf_by_tag.clear()
        self._backward_area_pdf_by_tag.clear()
        self._backward_jacobian_by_tag.clear()
        self._async_bdpt_next_segment_subpath = 0
        self._camera_sample_batch_id = 0
        self._async_forward_strike_count = 0
        self._async_forward_lens_hit_count = 0
        self._async_forward_lens_hit_after_bounce_count = 0
        self._async_forward_launched_count = 0
        self._async_backward_attempt_count = 0
        self._async_backward_parametric_absorbed_count = 0
        self._async_backward_launched_count = 0
        self._async_backward_strike_count = 0
        self.bdpt_last_shadow_stats = {}
        self.bdpt_last_optical_transfer = {}
        self.bdpt_last_camera_samples = {}
        self._bdpt_sensor_cfg = None

        cfg = lenses[idx]
        fp_x = self.focal_plane_x
        sensor_x = float(getattr(getattr(self.scene, "image_plate", None), "x", 0.0))
        defocus_mm = (fp_x - sensor_x) * 1000.0 if math.isfinite(fp_x) else float("nan")
        print(
            f"[lens-live] element {idx}/{len(lenses)-1}"
            f"  dx={delta_x*1000:+.3f}mm"
            f"  center_x={cfg.center_x:.5f}"
            f"  span=[{cfg.x_front:.5f}, {cfg.x_back:.5f}]"
            f"  focal_plane={fp_x:.5f}"
            f"  defocus={defocus_mm:+.2f}mm",
            flush=True,
        )

        # Fast focus probe: trace a ray bundle to measure actual CoC and true focus.
        optics = getattr(getattr(self, "_lens_assembly", None), "optics", None)
        if optics is not None:
            _ap_r = float(getattr(lenses[0], "aperture_radius", 0.015)) if lenses else 0.015
            _iris = getattr(self.scene, "iris_aperture", None)
            if _iris is not None and bool(getattr(_iris, "enabled", False)):
                _ap_r = min(_ap_r, float(getattr(_iris, "r_inner", _ap_r)))
            _probe = _probe_focus_coc(
                optics,
                float(self.scene.object_plane.x),
                float(getattr(getattr(self.scene, "image_plate", None), "x", 0.0)),
                _ap_r,
            )
            self._focus_probe_result = _probe
            print(
                f"[focus-probe] traced_focus={_probe['min_coc_x']:.5f}"
                f"  min_coc={_probe['min_coc_mm']:.3f}mm"
                f"  coc@sensor={_probe['coc_at_sensor_mm']:.3f}mm"
                f"  ({_probe['n_passed']} rays passed)",
                flush=True,
            )

    @property
    def focal_plane_x(self) -> float:
        """Design focal plane x — where the solver placed focus relative to the sensor.

        Uses the optical design solver's thin-lens group model, which is the same
        computation used to place the sensor, so the result is within sensor_error
        (~0.02mm) of image_plate.x at startup.  After manual lens adjustments the
        orange probe ring (from _focus_probe_result) shows the new traced focus;
        this yellow-ring property stays at the design position for reference.

        Returns float('inf') if not computable.
        """
        fp = _solver_focal_plane_x(self.scene)
        if math.isfinite(fp):
            return fp
        # Fallback: compound-lens paraxial matrix (may differ from solver's model)
        optics = getattr(getattr(self, "_lens_assembly", None), "optics", None)
        if optics is None:
            return float("inf")
        try:
            return _paraxial_image_x(optics, float(self.scene.object_plane.x))
        except Exception:
            return float("inf")

    def _start_progressive_refinement(self) -> None:
        """Start parametric→LUT→MLP background refinement if in PARAMETRIC mode."""
        if self._lens_assembly is None:
            return
        if self._lens_assembly.mode != LensAssemblySpec.MODE_PARAMETRIC:
            return
        if self._lens_assembly.optics is None:
            return
        self._lens_assembly.start_progressive_refinement()

    def _poll_assembly_mode_switch(self) -> None:
        """Apply any pending LUT/MLP transition from the background worker.
        Call once per render frame on the main thread."""
        if self._lens_assembly is None:
            return
        result = self._lens_assembly.poll_pending_mode_switch(self.tracer)
        if result == "LUT":
            self.register_lut_ctx_if_needed()
        elif result == "MLP":
            pass  # _register_mlp already re-registered triangles

    def register_lut_ctx_if_needed(self) -> None:
        """Register the LUT scale context if assembly switched to LUT mode."""
        if self._lens_assembly is None:
            return
        if self._lens_assembly._transfer_grid is None:
            return
        self._lens_assembly.register_lut_ctx(self.tracer)

    def _register_neural_payload(
        self,
        fwd_payload: np.ndarray,
        bwd_payload: Optional[np.ndarray] = None,
    ) -> None:
        """Load MLP payloads and register NEURAL_ASSEMBLY groups on the lens assembly."""
        if self._lens_assembly is None:
            self._lens_assembly = LensAssemblySpec()
        self._lens_assembly.load_payload(fwd_payload, bwd_payload)
        self._do_register_neural_assembly_group()

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
                _bdpt_compact_to_legacy_records(self._last_bdpt_records),
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

        verts_by_class: List[np.ndarray] = [_empty4, _empty4, _empty4, _empty4, _empty4, _empty4]
        n_by_class = [0, 0, 0, 0, 0, 0]

        if buf is not None and buf.shape[0] > 0:
            # Normalise world positions → [0,1] scene box for the renderer.
            norm_x = np.clip((buf[:, 0] - x_min) / x_span, 0.0, 1.0)
            norm_y = np.clip((buf[:, 1] + r)      / yz_span, 0.0, 1.0)
            norm_z = np.clip((buf[:, 2] + r)      / yz_span, 0.0, 1.0)
            amp_v  = buf[:, 3]
            cls_v  = buf[:, 4].astype(np.int32)
            if buf.shape[1] >= 6:
                gid_v = buf[:, 5].astype(np.int32)
                valid_gid = gid_v >= 0
                if np.any(valid_gid):
                    uq, uc = np.unique(gid_v[valid_gid], return_counts=True)
                    self._gid_crossing_counts = dict(zip(uq.tolist(), uc.tolist()))
                else:
                    self._gid_crossing_counts = {}

                # Remap entrance/exit GID hits to display classes 4 (red) and 5 (green).
                # Classes 4/5 are always kept — no amplitude threshold applied to them.
                _asm = getattr(self, "_lens_assembly", None)
                if _asm is not None:
                    _ap = _asm.acceptance_params()
                    if _ap is not None:
                        _ent_gid = int(_ap["entrance_gid"])
                        _exit_gid = int(_ap["exit_gid"])
                        if _ent_gid >= 0:
                            cls_v = np.where(gid_v == _ent_gid, 4, cls_v)
                        if _exit_gid >= 0:
                            cls_v = np.where(gid_v == _exit_gid, 5, cls_v)

            # Amplitude threshold for volume/field points only (cls 1, 2).
            # Strike points (cls 0, 3, 4, 5) are always kept.
            # Use a fast sample-based estimate instead of a full sort — sorting
            # 20M elements per frame blocks the main thread long enough to
            # trigger Windows TDR and crash the GL context.
            strike_mask = (cls_v == 0) | (cls_v == 3) | (cls_v == 4) | (cls_v == 5)
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

            for cls in range(6):
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
            f"pts={[n_by_class[c] for c in range(6)]}",
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


_dm_font_cache: list = [None]


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

    # 10 cm scale stick at object plane (same in both views).
    _dm_col = (255, 185, 0)
    if top_view:
        _dm_p0 = map_top(scene, scene.object_plane.x, -0.05, rect)
        _dm_p1 = map_top(scene, scene.object_plane.x, +0.05, rect)
    else:
        _dm_p0 = map_side(scene, scene.object_plane.x, -0.05, rect)
        _dm_p1 = map_side(scene, scene.object_plane.x, +0.05, rect)
    pygame.draw.line(surf, _dm_col, _dm_p0, _dm_p1, 2)
    _tk = 5
    pygame.draw.line(surf, _dm_col, (_dm_p0[0] - _tk, _dm_p0[1]), (_dm_p0[0] + _tk, _dm_p0[1]), 1)
    pygame.draw.line(surf, _dm_col, (_dm_p1[0] - _tk, _dm_p1[1]), (_dm_p1[0] + _tk, _dm_p1[1]), 1)
    if pygame.font.get_init():
        if _dm_font_cache[0] is None:
            _dm_font_cache[0] = (pygame.font.SysFont("consolas", 9) or
                                 pygame.font.SysFont("monospace", 9) or False)
        _sf = _dm_font_cache[0]
        if _sf:
            _lbl = _sf.render("10cm", True, _dm_col)
            surf.blit(_lbl, (_dm_p1[0] + _tk + 2, _dm_p1[1] - _lbl.get_height() // 2))


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
        auto_fit_view=scene.auto_fit_view,
        view_aspect=scene.view_aspect,
        enable_debug_plate=scene.enable_debug_plate,
        debug_plate_x=scene.debug_plate_x,
        debug_plate_radius=scene.debug_plate_radius,
        object_plane=scene.object_plane,
        image_plate=scene.image_plate,
        include_legacy_stage=scene.include_legacy_stage,
        subject_scene_mode=scene.subject_scene_mode,
        subject_time_s=scene.subject_time_s,
        subject_scale=scene.subject_scale,
        subject_depth_scale=scene.subject_depth_scale,
        subject_x=scene.subject_x,
        subject_y=scene.subject_y,
        subject_z=scene.subject_z,
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
        lens_hood_front_radius=scene.lens_hood_front_radius,
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
    plate_l = np.sum(plate.astype(np.float64), axis=2)
    throughput = float(np.sum(plate_l))
    if throughput <= EPS:
        return 1.0e6, {"records": int(records), "mse": float("inf"), "throughput": 0.0}
    mse = 0.0
    score = 0.010 / math.sqrt(max(1.0, throughput))
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
        plate = bench.trace_forward_backward_sensor_rgb(
            int(pixels),
            aperture_samples=1,
            seed=int(seed + 101),
            max_bounces=24,
        ).astype(np.float64)
    else:
        plate = bench.capture_plate_rgb(int(pixels)).astype(np.float64)
    plate_l = np.sum(plate, axis=2)
    pmax = float(np.max(plate_l))
    if pmax <= EPS:
        return 1.0e6
    plate_l = plate_l / pmax
    mse = 0.0
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
    sensor_res: int = 64,
    sensor_amp_gain: float = 1.0,
    sensor_min_amplitude: float = 0.0,
    emitter_amp_gain: float = 1.0,
    compute_mode: str = "gpu",
    profile: bool = False,
    field_capture: bool = True,
    bake_noodles: int = 0,
    focus_steps: int = 1,
    focus_range_mm: float = 2.0,
    bake_wavelengths: int = 3,
    bake_training_table: str = "",
    bake_training_gb: float = 0.0,
    neural_assembly: bool = False,
    neural_train_epochs: int = 60,
    neural_payload_in: str = "",
    neural_payload_out: str = "",
    parametric: bool = False,
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
            GL_DEPTH_TEST, GL_DEPTH_BUFFER_BIT, glDepthMask, glDepthFunc, GL_LEQUAL,
            GL_BLEND, GL_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
            glViewport,
            glEnableClientState, glDisableClientState, glVertexPointer, glDrawArrays,
            GL_VERTEX_ARRAY, GL_TRIANGLE_STRIP, GL_POINTS, GL_LINES, GL_LINE_STRIP,
            glPointSize, glLineWidth,
            glDeleteTextures, glActiveTexture, GL_TEXTURE0, GL_TEXTURE1,
            glCreateShader, glShaderSource, glCompileShader, glGetShaderiv,
            glGetShaderInfoLog, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
            GL_COMPILE_STATUS, glCreateProgram, glAttachShader, glLinkProgram,
            glGetProgramiv, glGetProgramInfoLog, GL_LINK_STATUS, glUseProgram,
            glGetUniformLocation, glUniform1i, glUniform1f, glUniform3f, glUniform4f, glDeleteProgram,
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
    scene.view_aspect = float(W) / float(max(1, H))
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
    bench.tracer.set_bdpt_sweep_trigger(_pip_res * _pip_res)
    if neural_payload_in:
        # ── Load pre-trained payload(s) and register immediately ──────────────
        _loaded_payload = np.load(neural_payload_in).astype(np.float32)
        print(f"[neural] loaded forward payload from {neural_payload_in} "
              f"({len(_loaded_payload)} floats)", flush=True)
        _bwd_path = neural_payload_in.replace(".npy", "_bwd.npy")
        _loaded_bwd = None
        if os.path.exists(_bwd_path):
            _loaded_bwd = np.load(_bwd_path).astype(np.float32)
            print(f"[neural] loaded backward payload from {_bwd_path} "
                  f"({len(_loaded_bwd)} floats)", flush=True)
        bench._register_neural_payload(_loaded_payload, _loaded_bwd)
    elif neural_assembly:
        # ── Full neural-assembly pipeline: bake → train → export → register ───
        import math as _math
        from numpy.lib.format import open_memmap as _open_memmap
        from camera_designer.neural_assembly import (
            train as _na_train,
            export_payload as _na_export,
            fit_acceptance_boundary as _na_fit_boundary,
        )

        _na_train_path  = bake_training_table or "neural_assembly_train.npy"
        _na_n_rays      = max(1, int(bake_noodles)) if int(bake_noodles) > 0 else 65_536
        _na_target_rows = max(1024, int(float(bake_training_gb) * (1024 ** 3) / 44))
        _na_n_wl        = int(bake_wavelengths)

        # Derive lens geometry from loaded scene
        _lsg = getattr(bench, "lens_surface_groups", [])
        if not _lsg:
            raise RuntimeError("[neural] No lens_surface_groups — upload_scene_data first")
        _lenses = _scene_lenses(bench.scene)
        if not _lenses:
            raise RuntimeError("[neural] Cannot derive lens x positions — no LensConfig in scene")
        _x_ent  = float(_lenses[0].x_front)
        _x_exit = float(_lenses[-1].x_back)
        _r_lens = float(_lenses[0].aperture_radius)

        _wls = [0.486, 0.587, 0.656][:_na_n_wl]

        _N_COLS = 11
        _table  = _open_memmap(_na_train_path, mode="w+", dtype=np.float32,
                               shape=(_na_target_rows, _N_COLS))

        _rng     = np.random.default_rng(42)
        _written = 0
        _page    = 0
        _bnd_r:  list = []
        _bnd_dz: list = []
        _bnd_ok: list = []

        while _written < _na_target_rows:
            _wl   = float(_wls[_page % len(_wls)])
            _seed = 42 + _page
            _page += 1

            # Uniform disc at entrance plane; stochastic hemisphere into lens (-X)
            _r2    = _rng.uniform(0.0, _r_lens ** 2, _na_n_rays)
            _ang   = _rng.uniform(0.0, 2.0 * _math.pi, _na_n_rays)
            _r_in  = np.sqrt(_r2)
            _bake_o = np.column_stack([
                np.full(_na_n_rays, _x_ent + 1e-4, dtype=np.float64),
                (_r_in * np.cos(_ang)).astype(np.float64),
                (_r_in * np.sin(_ang)).astype(np.float64),
            ])
            _phi_d  = _rng.uniform(0.0, 2.0 * _math.pi, _na_n_rays)
            _cos_th = _rng.random(_na_n_rays)
            _sin_th = np.sqrt(1.0 - _cos_th ** 2)
            _bake_d = np.column_stack([
                -_cos_th,
                _sin_th * np.cos(_phi_d),
                _sin_th * np.sin(_phi_d),
            ]).astype(np.float64)

            _bnd_r.append(_r_in.astype(np.float32))
            _bnd_dz.append(_cos_th.astype(np.float32))

            recs = bench.trace_forward(
                0, _seed,
                max_bounces=20,
                blocking=True,
                bake_origins=_bake_o,
                bake_directions=_bake_d,
            )

            if len(recs.get("kind", [])) == 0:
                _bnd_ok.append(np.zeros(_na_n_rays, dtype=bool))
                continue

            _kind_a   = np.asarray(recs["kind"])
            _tag_a    = np.asarray(recs["tag"])
            _pos_a    = np.asarray(recs["pos"])
            _dir_a    = np.asarray(recs["dir"])
            _seg_a    = np.asarray(recs["seg_start"])
            _plen_a   = np.asarray(recs["path_len"])
            _bounce_a = np.asarray(recs["bounce"])
            _sensor_a = np.asarray(recs["is_sensor"]).astype(bool)

            _keep     = (_kind_a == 0) | (_kind_a == 1)
            _tag_a    = _tag_a[_keep];    _pos_a    = _pos_a[_keep]
            _dir_a    = _dir_a[_keep];    _seg_a    = _seg_a[_keep]
            _plen_a   = _plen_a[_keep];   _bounce_a = _bounce_a[_keep]
            _sensor_a = _sensor_a[_keep]

            _entry_by_tag:  dict = {}
            _sensor_by_tag: dict = {}
            for _i in range(len(_tag_a)):
                _t = int(_tag_a[_i])
                if _bounce_a[_i] == 0 and _t not in _entry_by_tag:
                    _entry_by_tag[_t] = _i
                if _sensor_a[_i] and _t not in _sensor_by_tag:
                    _sensor_by_tag[_t] = _i

            _alive = np.zeros(_na_n_rays, dtype=bool)
            for _t in _sensor_by_tag:
                if _t < _na_n_rays:
                    _alive[_t] = True
            _bnd_ok.append(_alive)

            _rows = []
            for _t, _si in _sensor_by_tag.items():
                _ei = _entry_by_tag.get(_t)
                if _ei is None:
                    continue
                _, _ye, _ze      = _pos_a[_ei]
                _dxe, _dye, _dze = _dir_a[_ei]
                _theta_hit = float(np.arctan2(_ze, _ye))
                _r_in_v    = float(np.hypot(_ye, _ze))
                _cos_t     = _math.cos(_theta_hit)
                _sin_t     = _math.sin(_theta_hit)
                _dir_z_in   = float(-_dxe)
                _dir_r_in   = float( _dye * _cos_t + _dze * _sin_t)
                _dir_phi_in = float(-_dye * _sin_t + _dze * _cos_t)

                _, _yo, _zo      = _seg_a[_si]
                _dxo, _dyo, _dzo = _dir_a[_si]
                _theta_out  = float(np.arctan2(_zo, _yo))
                _r_out_v    = float(np.hypot(_yo, _zo))
                _delta_phi  = _theta_out - _theta_hit
                _delta_phi -= 2.0 * _math.pi * round(_delta_phi / (2.0 * _math.pi))
                _dir_r_out   = float( _dyo * _cos_t + _dzo * _sin_t)
                _dir_phi_out = float(-_dyo * _sin_t + _dzo * _cos_t)
                _dir_z_out   = float(-_dxo)
                _opl         = float(_plen_a[_si])

                _rows.append((_r_in_v, _dir_r_in, _dir_phi_in, _dir_z_in, _wl,
                              _r_out_v, _delta_phi, _dir_r_out, _dir_phi_out, _dir_z_out, _opl))

            if _rows:
                _chunk = np.array(_rows, dtype=np.float32)
                _take  = min(len(_chunk), _na_target_rows - _written)
                _table[_written:_written + _take] = _chunk[:_take]
                _written += _take

            print(f"[bake/gpu] {_written:,}/{_na_target_rows:,}"
                  f"  wl={_wl:.4f}µm  hits={len(_rows):,}", flush=True)

        _na_bnd = np.array([0.0, 0.0], np.float32)
        if _bnd_r:
            _c0, _c1 = _na_fit_boundary(
                np.concatenate(_bnd_r),
                np.concatenate(_bnd_dz),
                np.concatenate(_bnd_ok),
                r_lens=float(_r_lens),
            )
            _na_bnd = np.array([_c0, _c1], np.float32)
        _table.flush()
        print(f"[neural] training table {_na_train_path}: "
              f"{_written:,}/{_na_target_rows:,} rows  "
              f"boundary c0={float(_na_bnd[0]):.4f} c1={float(_na_bnd[1]):.4f}",
              flush=True)

        _na_kw = dict(
            z_entry=float(_x_ent),
            z_exit=float(_x_exit),
            r_lens=float(_r_lens),
            boundary_c0=float(_na_bnd[0]),
            boundary_c1=float(_na_bnd[1]),
        )

        # Train once — same weights serve both directions.
        # Backward surface uses the same payload with z_entry/z_exit swapped.
        _na_model, _na_norm = _na_train(
            _na_train_path, epochs=int(neural_train_epochs), verbose=True)
        _na_payload     = _na_export(_na_model, _na_norm, **_na_kw)
        _na_payload_bwd = _na_export(_na_model, _na_norm,
                                     z_entry=_na_kw["z_exit"],
                                     z_exit=_na_kw["z_entry"])

        if neural_payload_out:
            np.save(neural_payload_out, _na_payload)
            np.save(neural_payload_out.replace(".npy", "_bwd.npy"), _na_payload_bwd)
            print(f"[neural] saved payloads → {neural_payload_out} + _bwd", flush=True)

        bench._register_neural_payload(_na_payload, _na_payload_bwd)
    elif parametric:
        # ── Exact algebraic parametric lens: bypass LUT/MLP entirely ─────────
        _cl = _compound_lens_from_scene(bench.scene)
        print(
            f"[parametric] CompoundLens: {len(_cl.elements)} elements"
            f"  f_eff={_cl.f_eff * 1e3:.1f} mm  f/{_cl.f_number:.1f}",
            flush=True,
        )

        if bench._lens_assembly is None:
            bench._lens_assembly = LensAssemblySpec()
        bench._lens_assembly.set_optics(_cl, mode=LensAssemblySpec.MODE_PARAMETRIC)

        lsg = getattr(bench, "lens_surface_groups", [])
        if lsg:
            bench._lens_assembly.register(
                bench.tracer,
                lsg,
                bench.tri_vertices,
                bench.tri_centroids,
                _scene_lenses(bench.scene),
            )
        else:
            print("[parametric] no lens_surface_groups — will register on first render",
                  flush=True)
    elif bake_training_gb > 0.0:
        raise RuntimeError("Standalone training table baking was removed; use the live pipeline camera path.")
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
            float a = (u_mode < 1.5) ? 0.0 : 1.57079632679;
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
            float a = (u_mode < 1.5) ? 0.0 : 1.57079632679;
            float ca = cos(a);
            float sa = sin(a);
            vec3 p = gl_Vertex.xyz;
            vec2 yz = p.yz - vec2(0.5);
            float v = dot(yz, vec2(ca, sa));
            gl_Position = vec4(p.x * 2.0 - 1.0, v * 2.0, 0.0, 1.0);
            // Keep diagnostic points subordinate to the exposure texture.
            gl_PointSize = 1.35;
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
            // 4=entrance-surface red  5=exit-surface green
            vec3 base;
            if      (u_class < 0.5) base = vec3(1.00, 0.60, 0.18);
            else if (u_class < 1.5) base = vec3(0.18, 0.88, 1.00);
            else if (u_class < 2.5) base = vec3(0.18, 1.00, 0.45);
            else if (u_class < 3.5) base = vec3(1.00, 0.28, 0.72);
            else if (u_class < 4.5) base = vec3(1.00, 0.18, 0.18);   // entrance: hot red
            else                    base = vec3(0.18, 1.00, 0.18);   // exit: bright green
            // Entrance/exit hits glow brighter — boost and keep constant alpha
            float is_special = step(3.5, u_class);
            float amp_boost  = mix(1.0, 3.0, is_special);
            vec3 c = display_curve(vec3(v_amp * amp_boost)) * base;
            float alpha = mix(0.28, 0.55, is_special);
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

    # ── Acceptance-cone wireframe shader ─────────────────────────────────────
    # Simple 2-D pass-through: vertices arrive pre-projected in NDC.
    cone_vert_src = """
        #version 120
        void main() {
            gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);
        }
    """
    cone_frag_src = """
        #version 120
        uniform vec4 u_color;
        void main() {
            gl_FragColor = u_color;
        }
    """
    cvs = _compile_shader(GL_VERTEX_SHADER,   cone_vert_src)
    cfs = _compile_shader(GL_FRAGMENT_SHADER,  cone_frag_src)
    cone_prog = glCreateProgram()
    glAttachShader(cone_prog, cvs)
    glAttachShader(cone_prog, cfs)
    glLinkProgram(cone_prog)
    if not glGetProgramiv(cone_prog, GL_LINK_STATUS):
        msg = glGetProgramInfoLog(cone_prog)
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
            float vd = dot(yz, vec2(-sa, ca));  // depth: perp to view direction in YZ
            gl_Position = vec4(a_pos.x * 2.0 - 1.0, vp * 2.0, clamp(vd * 2.0, -1.0, 1.0), 1.0);
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
    surface_verts_by_class: List[np.ndarray] = [_empty4, _empty4, _empty4, _empty4, _empty4, _empty4]
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

    tex_forward_pip = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_forward_pip)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _pip_res, _pip_res, 0,
                 GL_RGB, GL_FLOAT, _pip_blank)

    tex_bdpt_pip = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_bdpt_pip)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
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
        # Use the actual scene material IDs so that emissive source surfaces
        # carry their real emission_rgb into the fragment
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
        # any angle; the sensor plate remains visible from the field side.
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
        # Let the renderer derive lights from all emissive source surfaces in
        # the scene; there is no manual light setup.
        _gl_renderer.derive_emissive_area_lights(verts8, mat_v, gid_v, min_emitter_group_id=-999)
        print(f"[gl-renderer] scene VAO built: {n_tris*3} verts, "
              f"{len(bank.groups)} UV groups", flush=True)

    # HUD text texture — RGBA8, sized to a reasonable stats strip width × height
    _HUD_W, _HUD_H = 380, 118
    tex_hud = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_hud)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    _hud_blank = np.zeros((_HUD_H, _HUD_W, 4), dtype=np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _HUD_W, _HUD_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, _hud_blank)

    # PIP viewports: bdpt image + reverse sensor feed + forward strike image.
    # Three pips left-to-right: [bdpt | backward | forward]
    _pip_dim = int(min(W * 0.22, H * 0.22))
    _pip_gap = 10
    _bdpt_pip_vx = (W - (3 * _pip_dim + 2 * _pip_gap)) // 2
    _pip_vx      = _bdpt_pip_vx + _pip_dim + _pip_gap
    _uv_pip_vx   = _pip_vx      + _pip_dim + _pip_gap
    _pip_vy  = 6

    # Lazy font for PIP stats overlay – created on first draw to avoid init cost.
    _pip_font: list = [None]  # mutable container so closure can write it
    _u_border      = glGetUniformLocation(pip_prog, "u_border")
    _u_border_col  = glGetUniformLocation(pip_prog, "u_border_col")

    def _draw_quad_with_pip_prog(tex: int, vx: int, vy: int, vw: int, vh: int,
                                 hud_mode: bool = False,
                                 border_col: tuple[float, float, float] = (0.35, 0.85, 1.0)) -> None:
        """Draw a fullscreen quad in the given viewport using pip_prog."""
        glViewport(vx, vy, vw, vh)
        glUseProgram(pip_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glUniform1i(glGetUniformLocation(pip_prog, "u_pip"), 0)
        glUniform1f(_u_border, 1.0 if hud_mode else 0.0)
        glUniform3f(_u_border_col, *border_col)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, fullscreen_quad)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisable(GL_BLEND)
        glUseProgram(0)

    def draw_pip() -> None:
        # ── Left pip: empty (border only) ───────────────────────────────────
        _draw_quad_with_pip_prog(
            tex_bdpt_pip, _bdpt_pip_vx, _pip_vy, _pip_dim, _pip_dim,
            border_col=(0.3, 0.3, 0.3),
        )

        # ── Centre pip: BDPT sensor plate (cyan border) ─────────────────────
        bdpt_plate = bench._last_bdpt_plate_rgb
        if bdpt_plate is not None and bdpt_plate.shape[0] > 0:
            bdpt_disp = np.ascontiguousarray(bdpt_plate[::-1, :], dtype=np.float32)
            glBindTexture(GL_TEXTURE_2D, tex_pip)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,
                         bdpt_disp.shape[1], bdpt_disp.shape[0], 0,
                         GL_RGB, GL_FLOAT, bdpt_disp)
        _draw_quad_with_pip_prog(tex_pip, _pip_vx, _pip_vy, _pip_dim, _pip_dim)

        # ── Right pip: forward ray strikes (orange border) ──────────────────
        fwd_img = bench.get_forward_strike_image()
        if fwd_img.shape[0] > 0:
            glBindTexture(GL_TEXTURE_2D, tex_forward_pip)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,
                         fwd_img.shape[1], fwd_img.shape[0], 0,
                         GL_RGB, GL_FLOAT, fwd_img)
        _draw_quad_with_pip_prog(
            tex_forward_pip, _uv_pip_vx, _pip_vy, _pip_dim, _pip_dim,
            border_col=(1.00, 0.62, 0.18),
        )

        # --- BDPT stats text rendered as a texture quad above the PIP ---
        try:
            stats = bench.tracer.get_bdpt_stats()
        except Exception:
            stats = {}
        if _pip_font[0] is None:
            pygame.font.init()
            _pip_font[0] = (pygame.font.SysFont("consolas", 11) or
                            pygame.font.SysFont("monospace", 11))
        font = _pip_font[0]
        nd = stats.get("nearest_dist_m", -1.0) if isinstance(stats, dict) else -1.0
        bc = stats.get("best_collinearity", 0.0) if isinstance(stats, dict) else 0.0
        dist_str = (f"dyz {nd*1000:.2f}mm" if nd >= 0.0 else "dyz --")
        mode_str = "pipeline-camera"
        line_h = font.get_linesize()

        def _render_hud_lines(lines, colours):
            surf = pygame.Surface((_HUD_W, _HUD_H), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            max_lines = max(1, _HUD_H // max(1, line_h + 1))
            for idx, txt in enumerate(lines[:max_lines]):
                col = colours[idx] if idx < len(colours) else (80, 210, 255)
                surf.blit(font.render(txt, True, col), (2, idx * (line_h + 1)))
            raw_a  = pygame.surfarray.array_alpha(surf.convert_alpha())
            raw_c  = pygame.surfarray.array3d(surf)
            arr    = np.zeros((_HUD_H, _HUD_W, 4), dtype=np.uint8)
            arr[:, :, :3] = np.transpose(raw_c, (1, 0, 2))
            arr[:, :,  3] = np.transpose(raw_a, (1, 0))
            return arr[::-1].copy()

        def _draw_hud(lines, colours, vx):
            rgba = _render_hud_lines(lines, colours)
            glBindTexture(GL_TEXTURE_2D, tex_hud)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _HUD_W, _HUD_H, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, rgba)
            _draw_quad_with_pip_prog(
                tex_hud,
                int(vx),
                int(_pip_vy + _pip_dim + 2),
                _HUD_W,
                _HUD_H,
                hud_mode=True,
            )

        # Pipeline camera label above left (violet) PIP.
        _draw_hud([mode_str], [(180, 140, 255)], _bdpt_pip_vx)

        # ── Backward transport stats above center (cyan) PIP ───────────────
        _sh_stats   = bench.bdpt_last_shadow_stats
        _sh_cands   = int(_sh_stats.get("candidates",  0))
        _sh_tested  = int(_sh_stats.get("shadow_rays", 0))
        _sh_clear   = int(_sh_stats.get("visible",     0))
        _sh_blocked = int(_sh_stats.get("blocked",     0))
        _sh_n_fwd   = int(_sh_stats.get("n_fwd",       0))
        _sh_n_bwd   = int(_sh_stats.get("n_bwd",       0))
        _sh_n_valid = int(_sh_stats.get("n_bwd_valid", 0))
        _sh_n_pairs = int(_sh_stats.get("n_pairs",     0))
        _sh_d_vis   = int(_sh_stats.get("direct_vis",   0))
        _sh_d_tot   = int(_sh_stats.get("direct_total", 0))
        _sh_den     = max(1, _sh_tested)
        _sh_clear_pct   = 100.0 * _sh_clear   / _sh_den
        _sh_blocked_pct = 100.0 * _sh_blocked / _sh_den
        # Stage label: show the first stage that is 0 to pinpoint the bottleneck.
        if _sh_n_fwd == 0:
            _stage_lbl = "STALL:no-fwd-records"
        elif _sh_n_bwd == 0:
            _stage_lbl = "STALL:no-bwd-records"
        elif _sh_n_valid == 0:
            _stage_lbl = "STALL:bwd-pix-cap-drop"
        elif _sh_n_pairs == 0:
            _stage_lbl = "STALL:no-band-overlap"
        elif int(_sh_stats.get("n_valid_dist", -1)) == 0:
            _stage_lbl = "STALL:pairs-invalid-dist"
        elif _sh_tested == 0:
            _stage_lbl = "shadow-in-progress"
        else:
            _stage_lbl = f"shadow {_sh_tested:_}"
        _d_pct = 100.0 * _sh_d_vis / max(1, _sh_d_tot)
        tracking_lines = [
            f"{dist_str} col {bc:.3f}",
            f"async fwd {bench._async_bdpt_forward_count()} bwd {bench._async_bdpt_backward_count()}",
            f"bwd try {bench._async_backward_attempt_count:_} abs {bench._async_backward_parametric_absorbed_count:_}",
            f"cam samp {int(bench.bdpt_last_camera_samples.get('samples', 0)):_} pdf {float(bench.bdpt_last_camera_samples.get('strategy_pdf_mean', 0.0)):.1e}",
            f"optic ev {int(bench.bdpt_last_optical_transfer.get('events', 0)):_} fail {int(bench.bdpt_last_optical_transfer.get('failed_rays', 0)):_}",
            f"optic top e{bench.bdpt_last_optical_transfer.get('top_fail_element', -1)} {bench.bdpt_last_optical_transfer.get('top_fail_reason', '')}",
            f"endpts fwd {_sh_n_fwd} bwd {_sh_n_bwd} valid {_sh_n_valid}",
            f"pairs {_sh_n_pairs}  {_stage_lbl}",
            f"clear {_sh_clear:_} ({_sh_clear_pct:.1f}%)  blk {_sh_blocked_pct:.1f}%",
            f"direct {_sh_d_vis:_}/{_sh_d_tot:_} ({_d_pct:.1f}% lit)",
        ]
        _stall = _sh_tested == 0 and _sh_d_vis == 0
        tracking_cols  = [(80, 210, 255)] * 3 + [
            (120, 255, 160) if int(bench.bdpt_last_camera_samples.get("samples", 0)) > 0 else (160, 160, 160),
            (255, 180, 90) if int(bench.bdpt_last_optical_transfer.get("failed_rays", 0)) > 0 else (120, 255, 160),
            (255, 150, 90) if int(bench.bdpt_last_optical_transfer.get("failed_rays", 0)) > 0 else (160, 160, 160),
            (200, 200, 100),
            (255, 80, 80) if (_sh_tested == 0) else (120, 255, 160),
            (255, 160, 100),
            (100, 220, 255) if _sh_d_vis > 0 else (160, 160, 160),
        ]
        _draw_hud(tracking_lines, tracking_cols, _pip_vx)

        # ── Camera assembly relaxation stats above right (orange) PIP ──────
        _gcc = bench._gid_crossing_counts
        _asm = bench._lens_assembly
        if _asm is not None:
            try:
                _asm.refresh_teleport_stats(bench.tracer)
            except Exception:
                pass
        _od = getattr(scene, "optical_design", None)
        _lenses = [l for l in (getattr(scene, "lens_stack", []) or []) if l is not None]
        if _lenses:
            _front = float(_lenses[0].x_front)
            _back = float(_lenses[-1].x_back)
            _gap_vals = [float(b.x_front - a.x_back) for a, b in zip(_lenses, _lenses[1:])]
            _min_gap_mm = (min(_gap_vals) * 1e3) if _gap_vals else 0.0
            _max_r_mm = max(float(l.aperture_radius) for l in _lenses) * 1e3
            _span_mm = (_back - _front) * 1e3
        else:
            _front = _back = _min_gap_mm = _max_r_mm = _span_mm = 0.0
        _target_f = float(getattr(getattr(_od, "spec", None), "target_focal_length_m", 0.0))
        _eff_f = float(getattr(_od, "effective_focal_length_m", 0.0))
        _sensor_err = float(getattr(_od, "sensor_error_m", 0.0))
        _sensor_x = float(getattr(getattr(scene, "image_plate", None), "x", 0.0))
        _obj_x = float(getattr(getattr(scene, "object_plane", None), "x", 0.0))
        _iris = getattr(scene, "iris_aperture", None)
        _iris_x = float(getattr(_iris, "x_pos", 0.0)) if _iris is not None else 0.0
        _iris_r = float(getattr(_iris, "r_inner", 0.0)) if _iris is not None else 0.0
        _ep_x = float(getattr(scene, "exit_pupil_x", 0.0))
        _ep_r = float(getattr(scene, "exit_pupil_radius", 0.0))
        _mode = getattr(_asm, "mode", "none") if _asm is not None else "none"
        _ent_gid = int(getattr(_asm, "_entrance_gid", -1)) if _asm is not None else -1
        _ex_gid = int(getattr(_asm, "_exit_gid", -1)) if _asm is not None else -1
        _n_ent = int(_gcc.get(_ent_gid, 0)) if _ent_gid >= 0 else 0
        _n_ex = int(_gcc.get(_ex_gid, 0)) if _ex_gid >= 0 else 0
        _ent_stats = getattr(_asm, "_entrance_stats", {}) if _asm is not None else {}
        _ex_stats = getattr(_asm, "_exit_stats", {}) if _asm is not None else {}
        _fwd_t = int(_ent_stats.get("transmitted", 0) or 0)
        _fwd_a = int(_ent_stats.get("absorbed", 0) or 0)
        _bwd_t = int(_ex_stats.get("transmitted", 0) or 0)
        _bwd_a = int(_ex_stats.get("absorbed", 0) or 0)
        _fwd_den = max(1, _fwd_t + _fwd_a)
        _bwd_den = max(1, _bwd_t + _bwd_a)
        _fwd_pct = 100.0 * float(_fwd_t) / float(_fwd_den)
        _bwd_pct = 100.0 * float(_bwd_t) / float(_bwd_den)
        _field_pair = getattr(bench, "_optical_field_pair", None)
        if _field_pair is not None:
            _front_cut = float(_field_pair["front"].cutoff_half_angle_rad) * 180.0 / math.pi
            _back_cut = float(_field_pair["back"].cutoff_half_angle_rad) * 180.0 / math.pi if "back" in _field_pair else 0.0
        else:
            _front_cut = _back_cut = 0.0
        assembly_lines = [
            f"cam {_mode} pipeline",
            f"f {(_eff_f*1e3):.1f}/{(_target_f*1e3):.1f}mm err {(_sensor_err*1e3):+.2f}mm",
            f"x obj {_obj_x:.3f} g1 {_front:.3f} sens {_sensor_x:.3f}",
            f"span {_span_mm:.1f}mm gap {_min_gap_mm:.1f}mm r {_max_r_mm:.1f}mm",
            f"iris {_iris_x:.3f} r{_iris_r*1e3:.1f} ep {_ep_x:.3f} r{_ep_r*1e3:.1f}",
            f"fwd T/A {_fwd_t}/{_fwd_a} {_fwd_pct:.1f}% ent_gid {_ent_gid} hits {_n_ent}",
            f"bwd T/A {_bwd_t}/{_bwd_a} {_bwd_pct:.1f}% ex_gid {_ex_gid} hits {_n_ex}",
            f"cut f/b {_front_cut:.1f}/{_back_cut:.1f}deg",
        ]
        assembly_cols = [(255, 170, 70)] * len(assembly_lines)
        _draw_hud(assembly_lines, assembly_cols, _uv_pip_vx)

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
    _CLASS_NAMES = ["fwd-strike", "fwd-volume", "rev-volume", "rev-strike", "entrance", "exit"]

    def draw_surface_points(mode: int, gain: float, t_now: float) -> None:
        any_pts = any(v.shape[0] > 0 for v in _svc)
        if not any_pts:
            return
        glUseProgram(point_prog)
        glUniform1f(glGetUniformLocation(point_prog, "u_time"), float(t_now))
        glUniform1f(glGetUniformLocation(point_prog, "u_mode"), float(mode))
        glUniform1f(glGetUniformLocation(point_prog, "u_gain"), float(gain))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
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

    def draw_acceptance_cones(t_now: float, mode: int = 0) -> None:
        """Draw wireframe acceptance cones for the registered lens assembly.

        Renders two aperture circles (at entrance and exit planes) plus the four
        generator lines that bound the forward and backward acceptance cones.
        Everything is in normalised [0,1]³ scene space with the same Y/Z rotation
        as the point/volume shaders so all overlays stay co-registered.

        Colors: per-element front-side cones             = red
                per-element back-side cones              = green
                system front->back teleport boundary     = amber
                system back->front teleport boundary     = cyan
        """
        _asm = getattr(bench, "_lens_assembly", None)
        if _asm is None:
            return
        _ap = _asm.acceptance_params()
        if _ap is None:
            return
        try:
            _field_pair = _asm.profile_field_pair(
                object_point=(float(scene.object_plane.x), 0.0, 0.0),
                image_point=(float(scene.image_plate.x), 0.0, 0.0),
                verify=False,
            )
        except Exception:
            _field_pair = getattr(bench, "_optical_field_pair", None)
        _boundary_profiles = None
        try:
            _boundary_profiles = _asm.boundary_teleport_profiles(verify=True, n_azimuth=16)
        except Exception:
            if _field_pair is not None:
                _boundary_profiles = {
                    k: v.boundary for k, v in _field_pair.items()
                    if getattr(v, "boundary", None) is not None
                }

        x_span  = max(1e-8, float(scene.x_max - scene.x_min))
        yz_span = max(1e-8, float(2.0 * scene.view_radius))
        x_min_s = float(scene.x_min)

        def _nx(x):   return float((x - x_min_s) / x_span)
        def _nr(rad): return float(rad / yz_span)

        # entrance_x / exit_x are exact analytical surface vertex positions in
        # scene space — from optics.side("front"/"back").x_pos, which is correctly
        # offset by iris_x − aperture_z_local at CompoundLens construction time.
        x_ent_w  = float(_ap["entrance_x"])
        x_exit_w = float(_ap["exit_x"])
        x_ent    = _nx(x_ent_w)
        x_exit   = _nx(x_exit_w)
        r_ent    = _nr(float(_ap["entrance_r"]))
        r_exit   = _nr(float(_ap["exit_r"]))

        # True optical entrance pupil: image of aperture stop formed by front
        # elements.  x_ep is distinct from x_ent (physical front surface vertex).
        _x_ep_w = x_ent_w  # fallback: front surface
        if hasattr(_asm, "optics") and _asm.optics is not None:
            try:
                _x_ep_w = float(_asm.optics.entrance_pupil[0])
            except Exception:
                pass
        _x_ep = _nx(_x_ep_w)

        # Build aperture circle vertices (in normalised XYZ, Y/Z are transverse).
        # These mark the physical clear aperture at each side.
        N_SEG = 48
        angles = np.linspace(0.0, 2.0 * math.pi, N_SEG, endpoint=False, dtype=np.float32)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Entrance circle (red): XYZ columns, closed by repeating first point
        ent_circle = np.empty((N_SEG + 1, 3), dtype=np.float32)
        ent_circle[:N_SEG, 0] = x_ent
        ent_circle[:N_SEG, 1] = 0.5 + r_ent * cos_a
        ent_circle[:N_SEG, 2] = 0.5 + r_ent * sin_a
        ent_circle[N_SEG]     = ent_circle[0]

        # Exit circle (green)
        exit_circle = np.empty((N_SEG + 1, 3), dtype=np.float32)
        exit_circle[:N_SEG, 0] = x_exit
        exit_circle[:N_SEG, 1] = 0.5 + r_exit * cos_a
        exit_circle[:N_SEG, 2] = 0.5 + r_exit * sin_a
        exit_circle[N_SEG]     = exit_circle[0]

        # Characteristic optical length — purely from the lens assembly, not the mesh.
        # f_eff sets the natural scale; axial_depth is the fallback for degenerate lenses.
        axial_depth   = abs(x_exit_w - x_ent_w)
        f_eff_abs     = abs(float(_asm.optics.f_eff)) if hasattr(_asm, "optics") and _asm.optics is not None else axial_depth
        optical_scale = max(f_eff_abs, axial_depth, 1e-6)
        # Extend 8× the optical scale on each side — lens-derived, scene-agnostic.
        # Lines will be clipped at the screen edge if they run past the scene bounds.
        extent = optical_scale * 8.0

        # Object-side cone (red): apex at entrance, opens BACKWARD into the scene.
        # A ray from the scene within fwd_half_angle of the axis traverses the full column.
        x_obj_far_w = x_ent_w - extent
        x_obj_far   = _nx(x_obj_far_w)
        obj_half = float(_ap.get("fwd_half_angle", _ap.get("verified_spread_half_angle_rad", 0.0)))
        if _field_pair is not None:
            obj_half = max(obj_half, float(_field_pair["front"].cutoff_half_angle_rad))
        h_obj       = float(math.tan(obj_half) * extent / yz_span)
        obj_lines = np.array([
            [x_ent, 0.5,       0.5      ], [x_obj_far, 0.5 + h_obj, 0.5      ],
            [x_ent, 0.5,       0.5      ], [x_obj_far, 0.5 - h_obj, 0.5      ],
            [x_ent, 0.5,       0.5      ], [x_obj_far, 0.5,          0.5 + h_obj],
            [x_ent, 0.5,       0.5      ], [x_obj_far, 0.5,          0.5 - h_obj],
        ], dtype=np.float32)

        # Image-side cone (green): apex at exit, opens FORWARD toward the sensor.
        # A backward sensor ray within bwd_half_angle can traverse back to the scene.
        x_img_far_w = x_exit_w + extent
        x_img_far   = _nx(x_img_far_w)
        img_half = float(_ap.get("bwd_half_angle", 0.0))
        if _field_pair is not None and "back" in _field_pair:
            img_half = max(img_half, float(_field_pair["back"].cutoff_half_angle_rad))
        h_img       = float(math.tan(img_half) * extent / yz_span)
        img_lines = np.array([
            [x_exit, 0.5,      0.5      ], [x_img_far, 0.5 + h_img, 0.5      ],
            [x_exit, 0.5,      0.5      ], [x_img_far, 0.5 - h_img, 0.5      ],
            [x_exit, 0.5,      0.5      ], [x_img_far, 0.5,          0.5 + h_img],
            [x_exit, 0.5,      0.5      ], [x_img_far, 0.5,          0.5 - h_img],
        ], dtype=np.float32)

        t     = t_now * 0.16
        if mode < 0 or mode == 0:
            a_rot = 0.24 * math.sin(t)
        elif mode == 2:
            a_rot = math.pi / 2.0
        else:
            a_rot = 0.0
        ca, sa = math.cos(a_rot), math.sin(a_rot)

        def _project(pts3):
            """Project (N,3) XYZ → (N,2) NDC using the same rotation as point shader."""
            x  = pts3[:, 0]
            yz = pts3[:, 1:] - 0.5
            v  = yz[:, 0] * ca + yz[:, 1] * sa
            return np.stack([x * 2.0 - 1.0, v * 2.0], axis=1).astype(np.float32)

        glUseProgram(cone_prog)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glLineWidth(2.0)
        glEnableClientState(GL_VERTEX_ARRAY)

        def _draw_lines(pts3, r, g, b):
            pts2 = np.ascontiguousarray(_project(pts3))
            glUniform4f(glGetUniformLocation(cone_prog, "u_color"), r, g, b, 0.75)
            glVertexPointer(2, GL_FLOAT, 0, pts2)
            glDrawArrays(GL_LINES, 0, int(pts2.shape[0]))

        def _draw_loop(pts3, r, g, b):
            pts2 = np.ascontiguousarray(_project(pts3))
            glUniform4f(glGetUniformLocation(cone_prog, "u_color"), r, g, b, 0.75)
            glVertexPointer(2, GL_FLOAT, 0, pts2)
            glDrawArrays(GL_LINE_STRIP, 0, int(pts2.shape[0]))

        def _ring_at(center, radius):
            rad = _nr(float(radius))
            cx = _nx(float(center[0]))
            circle = np.empty((N_SEG + 1, 3), dtype=np.float32)
            circle[:N_SEG, 0] = cx
            circle[:N_SEG, 1] = 0.5 + rad * cos_a
            circle[:N_SEG, 2] = 0.5 + rad * sin_a
            circle[N_SEG] = circle[0]
            return circle

        def _boundary_generators(boundary):
            src = np.asarray(boundary.source_center, dtype=np.float64)
            dst = np.asarray(boundary.target_center, dtype=np.float64)
            src3 = np.array([_nx(src[0]), 0.5 + src[1] / yz_span, 0.5 + src[2] / yz_span], dtype=np.float32)
            tx = _nx(float(dst[0]))
            tr = _nr(float(boundary.target_radius))
            lines = []
            for y, z in ((tr, 0.0), (-tr, 0.0), (0.0, tr), (0.0, -tr)):
                lines.append(src3)
                lines.append(np.array([tx, 0.5 + y, 0.5 + z], dtype=np.float32))
            return np.asarray(lines, dtype=np.float32)

        def _face_circle(face_profile):
            rad = _nr(float(face_profile.face.radius))
            cx = _nx(float(face_profile.center[0]))
            circle = np.empty((N_SEG + 1, 3), dtype=np.float32)
            circle[:N_SEG, 0] = cx
            circle[:N_SEG, 1] = 0.5 + rad * cos_a
            circle[:N_SEG, 2] = 0.5 + rad * sin_a
            circle[N_SEG] = circle[0]
            return circle

        def _face_generators(face_profile):
            fp = np.asarray(face_profile.focal_point, dtype=np.float64)
            fp3 = np.array([_nx(fp[0]), 0.5 + fp[1] / yz_span, 0.5 + fp[2] / yz_span], dtype=np.float32)
            cx = _nx(float(face_profile.center[0]))
            rr = _nr(float(face_profile.face.radius))
            lines = []
            for y, z in ((rr, 0.0), (-rr, 0.0), (0.0, rr), (0.0, -rr)):
                lines.append(fp3)
                lines.append(np.array([cx, 0.5 + y, 0.5 + z], dtype=np.float32))
            return np.asarray(lines, dtype=np.float32)

        # Physical iris / aperture stop: fetched here so the bowtie below can use it.
        _iris_overlay = getattr(scene, "iris_aperture", None)

        # Sensor FOV bowtie: marginal rays from sensor edge through aperture stop edge.
        # When the iris is present, lines graze its rim and continue with the same slope
        # into the scene (crossing the axis beyond the lens — correct bowtie geometry).
        # Fallback: chief-ray bowtie converging to the entrance pupil axis point.
        _r_s   = float(getattr(getattr(scene, "image_plate", None), "radius", 0.0))
        _x_sen = float(getattr(getattr(scene, "image_plate", None), "x",      0.0))
        _fov_half = obj_half
        if hasattr(_asm, "optics") and _asm.optics is not None and _r_s > 1e-9:
            _f_e = abs(float(_asm.optics.f_eff))
            if _f_e > 1e-9:
                _fov_half = math.atan2(_r_s, _f_e)
        _h_fov = float(math.tan(_fov_half) * extent / yz_span)
        if _r_s > 1e-9 and _x_sen > 1e-9:
            _x_s_n = _nx(_x_sen)
            _h_s_n = _r_s / yz_span
            _iris_ok = (_iris_overlay is not None
                        and bool(getattr(_iris_overlay, "enabled", False))
                        and abs(_nx(float(_iris_overlay.x_pos)) - _x_s_n) > 1e-6)
            if _iris_ok:
                _x_pinch   = _nx(float(_iris_overlay.x_pos))
                _r_pinch_n = _nr(float(_iris_overlay.r_inner))
                _ddx   = _x_pinch - _x_s_n
                _slope = (_r_pinch_n - _h_s_n) / _ddx if abs(_ddx) > 1e-9 else 0.0
                _h_far = _r_pinch_n + _slope * (x_obj_far - _x_pinch)
                _fov_bowtie = np.array([
                    [_x_s_n, 0.5 + _h_s_n, 0.5],       [_x_pinch, 0.5 + _r_pinch_n, 0.5],
                    [_x_pinch, 0.5 + _r_pinch_n, 0.5],  [x_obj_far, 0.5 + _h_far, 0.5],
                    [_x_s_n, 0.5 - _h_s_n, 0.5],       [_x_pinch, 0.5 - _r_pinch_n, 0.5],
                    [_x_pinch, 0.5 - _r_pinch_n, 0.5],  [x_obj_far, 0.5 - _h_far, 0.5],
                    [_x_s_n, 0.5, 0.5 + _h_s_n],       [_x_pinch, 0.5, 0.5 + _r_pinch_n],
                    [_x_pinch, 0.5, 0.5 + _r_pinch_n],  [x_obj_far, 0.5, 0.5 + _h_far],
                    [_x_s_n, 0.5, 0.5 - _h_s_n],       [_x_pinch, 0.5, 0.5 - _r_pinch_n],
                    [_x_pinch, 0.5, 0.5 - _r_pinch_n],  [x_obj_far, 0.5, 0.5 - _h_far],
                ], dtype=np.float32)
            else:
                _fov_bowtie = np.array([
                    [_x_s_n, 0.5 + _h_s_n, 0.5],  [_x_ep, 0.5, 0.5],
                    [_x_ep, 0.5, 0.5],              [x_obj_far, 0.5 + _h_fov, 0.5],
                    [_x_s_n, 0.5 - _h_s_n, 0.5],  [_x_ep, 0.5, 0.5],
                    [_x_ep, 0.5, 0.5],              [x_obj_far, 0.5 - _h_fov, 0.5],
                    [_x_s_n, 0.5, 0.5 + _h_s_n],  [_x_ep, 0.5, 0.5],
                    [_x_ep, 0.5, 0.5],              [x_obj_far, 0.5, 0.5 + _h_fov],
                    [_x_s_n, 0.5, 0.5 - _h_s_n],  [_x_ep, 0.5, 0.5],
                    [_x_ep, 0.5, 0.5],              [x_obj_far, 0.5, 0.5 - _h_fov],
                ], dtype=np.float32)
            _draw_lines(_fov_bowtie, 0.88, 0.88, 0.88)

        # 10 cm scale stick at object plane (orange), perpendicular to axis.
        # Gives a real-world reference to judge how wide the FOV cone actually is.
        _x_op_n = _nx(float(scene.object_plane.x))
        _dm_h   = 0.05 / yz_span   # half of 10 cm in normalised coords
        _tick_w = max(0.004, 0.008 / max(1e-8, x_span))
        _dm_bar_y = np.array([
            [_x_op_n, 0.5 - _dm_h, 0.5],
            [_x_op_n, 0.5 + _dm_h, 0.5],
        ], dtype=np.float32)
        _dm_bar_z = np.array([
            [_x_op_n, 0.5, 0.5 - _dm_h],
            [_x_op_n, 0.5, 0.5 + _dm_h],
        ], dtype=np.float32)
        _dm_ticks = np.array([
            [_x_op_n - _tick_w, 0.5 - _dm_h, 0.5], [_x_op_n + _tick_w, 0.5 - _dm_h, 0.5],
            [_x_op_n - _tick_w, 0.5 + _dm_h, 0.5], [_x_op_n + _tick_w, 0.5 + _dm_h, 0.5],
        ], dtype=np.float32)
        _draw_loop(_dm_bar_y,  1.0, 0.72, 0.0)
        _draw_loop(_dm_bar_z,  1.0, 0.72, 0.0)
        _draw_lines(_dm_ticks, 1.0, 0.72, 0.0)

        # Physical iris / aperture stop: blue.
        if _iris_overlay is not None and bool(getattr(_iris_overlay, "enabled", False)):
            _draw_loop(
                _ring_at([float(_iris_overlay.x_pos), 0.0, 0.0], float(_iris_overlay.r_inner)),
                0.12, 0.42, 1.0,
            )

        # Parametric glass profiles: draw the actual spherical front/back
        # meridians and edge joins so the display reads as glass, not boxes.
        _display_lenses = [l for l in (getattr(scene, "lens_stack", []) or [getattr(scene, "lens", None)]) if l is not None and _lens_is_valid(l)]
        for _lens in _display_lenses:
            _r_vals = np.linspace(-float(_lens.aperture_radius), float(_lens.aperture_radius), 80, dtype=np.float64)
            _r_abs = np.abs(_r_vals)
            _front_x = _lens_front_x(_lens, _r_abs)
            _back_x = _lens_back_x(_lens, _r_abs)
            _front_profile = np.stack([
                np.array([_nx(float(x)) for x in _front_x], dtype=np.float32),
                0.5 + (_r_vals / yz_span).astype(np.float32),
                np.full(_r_vals.shape, 0.5, dtype=np.float32),
            ], axis=1)
            _back_profile = np.stack([
                np.array([_nx(float(x)) for x in _back_x], dtype=np.float32),
                0.5 + (_r_vals / yz_span).astype(np.float32),
                np.full(_r_vals.shape, 0.5, dtype=np.float32),
            ], axis=1)
            _edge_lines = np.array([
                [_nx(float(_front_x[0])), 0.5 + float(_r_vals[0]) / yz_span, 0.5],
                [_nx(float(_back_x[0])),  0.5 + float(_r_vals[0]) / yz_span, 0.5],
                [_nx(float(_front_x[-1])), 0.5 + float(_r_vals[-1]) / yz_span, 0.5],
                [_nx(float(_back_x[-1])),  0.5 + float(_r_vals[-1]) / yz_span, 0.5],
            ], dtype=np.float32)
            _draw_loop(_front_profile, 0.72, 0.92, 1.0)
            _draw_loop(_back_profile, 0.72, 0.92, 1.0)
            _draw_lines(_edge_lines, 0.72, 0.92, 1.0)


        # ── Range finder: design focal plane ───────────────────────────────
        # Yellow ring = where the solver placed focus (matches sensor at startup).
        # When the ring coincides with the sensor disc (green), focus is correct.
        # Uses bench.focal_plane_x which calls _solver_focal_plane_x() — consistent
        # with the solver's thin-lens model and thus with the sensor placement.
        _fp_x = bench.focal_plane_x
        if math.isfinite(_fp_x) and float(scene.x_min) <= _fp_x <= float(scene.x_max) + 0.02:
            _x_fp_n  = _nx(_fp_x)
            _r_fp_m  = float(getattr(getattr(scene, "image_plate", None), "radius", 0.025))
            _r_fp_n  = _nr(_r_fp_m)
            _sensor_x = float(getattr(getattr(scene, "image_plate", None), "x", 0.0))
            _defocus_mm = (_fp_x - _sensor_x) * 1000.0
            # Ring at focal plane (yellow)
            _draw_loop(_ring_at([_fp_x, 0.0, 0.0], _r_fp_m), 1.0, 1.0, 0.15)
            # Cross-wire at focal plane center
            _cross_r = _r_fp_n * 0.45
            _fp_cross = np.array([
                [_x_fp_n, 0.5 - _cross_r, 0.5], [_x_fp_n, 0.5 + _cross_r, 0.5],
                [_x_fp_n, 0.5, 0.5 - _cross_r], [_x_fp_n, 0.5, 0.5 + _cross_r],
            ], dtype=np.float32)
            _draw_lines(_fp_cross, 1.0, 1.0, 0.15)
            # Defocus line from focal plane to sensor (dim white)
            if abs(_defocus_mm) > 0.01 and _x_ep > 0.0:
                _x_s_n = _nx(_sensor_x)
                _defocus_bar = np.array([
                    [_x_fp_n, 0.5, 0.5], [_x_s_n, 0.5, 0.5],
                ], dtype=np.float32)
                _draw_lines(_defocus_bar, 0.9, 0.9, 0.4)

        # ── Traced focus ring (orange) ──────────────────────────────────────
        # Shows the minimum circle-of-confusion plane found by _probe_focus_coc().
        # Distinct from the paraxial ring (yellow) — deviates when aberrations are
        # significant or when the marginal-focus doesn't match the paraxial estimate.
        _probe = getattr(bench, "_focus_probe_result", None)
        if _probe is not None and math.isfinite(float(_probe.get("min_coc_x", float("nan")))):
            _tf_x = float(_probe["min_coc_x"])
            if float(scene.x_min) <= _tf_x <= float(scene.x_max) * 1.5:
                _x_tf_n = _nx(_tf_x)
                _r_tf_m = float(getattr(getattr(scene, "image_plate", None), "radius", 0.025))
                _draw_loop(_ring_at([_tf_x, 0.0, 0.0], _r_tf_m), 1.0, 0.50, 0.05)
                # Tick marks on the ring cross-wire (orange, slightly shorter)
                _cr_tf = _nr(_r_tf_m) * 0.30
                _tf_cross = np.array([
                    [_x_tf_n, 0.5 - _cr_tf, 0.5], [_x_tf_n, 0.5 + _cr_tf, 0.5],
                    [_x_tf_n, 0.5, 0.5 - _cr_tf], [_x_tf_n, 0.5, 0.5 + _cr_tf],
                ], dtype=np.float32)
                _draw_lines(_tf_cross, 1.0, 0.50, 0.05)

        glLineWidth(1.0)
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
        glEnable(GL_DEPTH_TEST)
        glDepthMask(True)
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
        glDepthMask(False)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)

        # Drive BaseGLRenderer: fly-camera perspective when fly_mode is active,
        # otherwise use a fixed orthographic X/Y view.
        if _scene_vao[0] is not None:
            glEnable(GL_DEPTH_TEST)
            glDepthMask(True)
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
                ca  = 1.0
                sa  = 0.0
                mvp = np.array([
                    [ 2,       0,      0,    -1           ],
                    [ 0,    2*ca,   2*sa,    -(ca+sa)     ],
                    [ 0,   -2*sa,   2*ca,    (sa-ca)      ],
                    [ 0,       0,      0,    1            ],
                ], dtype=np.float32)
                mvp_col = np.ascontiguousarray(mvp.T.ravel(), dtype=np.float32)
                mv_col  = np.ascontiguousarray(np.eye(4, dtype=np.float32).ravel())
            vao_id, n_verts = _scene_vao[0][0], _scene_vao[0][1]
            _gl_renderer.draw_mesh(vao_id, n_verts, mvp_col, mv_col)
            glDepthMask(False)
            glDisable(GL_DEPTH_TEST)

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

    projection_axis = 1  # 1 = X/Y orthographic projection, 2 = X/Z projection
    auto_wiggle = False
    show_impact_points = False
    volume_gain = 1.0

    # ── Fly camera (F to toggle) ──────────────────────────────────────────── #
    # Positions are in the same [0,1]³ normalised space the VAO uses.
    # Start in front of the object plane looking along +X (the optical axis).
    # yaw=π/2 → forward = +X; the viewer sees the YZ cross-section — lenses
    # and subject objects appear as circles rather than as side-on slivers.
    fly_mode  = False
    fly_pos   = np.array([ 1.35, 0.5, 0.5], dtype=np.float64)
    fly_yaw   = -math.pi / 2     # yaw=−π/2 → forward = −X (same look direction as camera)
    fly_pitch = 0.0
    _FLY_SPEED = 0.6   # normalised units / second
    _FLY_SENS  = 0.20  # degrees per pixel of mouse movement

    # KPN back-pressure gate: block submit until C++ intent queue drains to below
    # capacity.  display_pipeline_records() still runs every frame so the viewer
    # stays live while the pipeline is catching up.
    _MAX_IN_FLIGHT = BDPT_PIPELINE_CAP

    # Pipeline camera render state.
    _pipeline_camera_busy = [False]
    _startup_iris_pending = [True]

    def _fire_pipeline_camera_render() -> None:
        if _pipeline_camera_busy[0]:
            return
        _pipeline_camera_busy[0] = True
        def _worker():
            try:
                bench.trace_forward_backward_sensor_rgb(
                    pixels=max(8, min(48, int(scene.image_plate.sensor_res))),
                    aperture_samples=8,
                    seed=20260522 + id(bench) % 0xFFFFFF,
                    max_bounces=48,
                    n_rays_bdpt=1_000_000,
                )
            except Exception as _me:
                print(f"[pipeline-camera] render error: {_me}", flush=True)
            finally:
                _pipeline_camera_busy[0] = False
        threading.Thread(target=_worker, daemon=True).start()

    def _trace(rpe: int, sd: int, mb: int):
        import time as _t
        if bench._async_bdpt_backward_count() > 0 or bench._async_bdpt_forward_count() > 0:
            _fire_pipeline_camera_render()
        while not closing.is_set() and bench.tracer.in_flight_count() > _MAX_IN_FLIGHT:
            _t.sleep(0.002)   # back off; let drain-loop consume Q_intent
        if closing.is_set():
            return
        with bench._trace_lock:
            fwd_have = int(bench._async_forward_launched_count)
            fwd_target = int(getattr(bench, "_async_bdpt_forward_warmup_target", 0))
            if fwd_have < fwd_target:
                warmup_batch = int(max(1, getattr(bench, "_async_bdpt_forward_warmup_batch", 500_000)))
                warmup_total = int(min(max(1, fwd_target - fwd_have), warmup_batch))
                warmup_rpe = max(int(rpe), max(1, warmup_total // max(1, int(bench.src_pos.shape[0]))))
                bench.trace_forward(warmup_rpe, sd, max_bounces=mb)
                if frame % 30 == 0:
                    print(
                        "[bdpt-forward-warmup]",
                        f"forward_launched={fwd_have:_}/{fwd_target:_}",
                        f"fwd_retained={bench._async_bdpt_forward_count():_}",
                        f"bwd_retained={bench._async_bdpt_backward_count():_}",
                        f"fwd_strikes={bench._async_forward_strike_count:_}",
                        f"lens_hits={bench._async_forward_lens_hit_count:_}",
                        f"lens_after_bounce={bench._async_forward_lens_hit_after_bounce_count:_}",
                        f"submit_rpe={warmup_rpe:_}",
                        flush=True,
                    )
                return
            bench.trace_forward(rpe, sd, max_bounces=mb)
            submitted_bwd = bench.trace_sensor_cast(rpe, sd ^ 0x5A5A, max_bounces=mb)
            if frame % 30 == 0:
                _s = bench.bdpt_last_shadow_stats
                print(
                    "[pipeline-trace]",
                    f"forward_launched={bench._async_forward_launched_count:_}",
                    f"backward_attempted={bench._async_backward_attempt_count:_}",
                    f"backward_absorbed={bench._async_backward_parametric_absorbed_count:_}",
                    f"backward_launched={bench._async_backward_launched_count:_}",
                    f"fwd_retained={bench._async_bdpt_forward_count():_}",
                    f"bwd_retained={bench._async_bdpt_backward_count():_}",
                    f"bwd_strikes={bench._async_backward_strike_count:_}",
                    f"submit_rpe={rpe:_}",
                    f"submit_bwd={int(submitted_bwd):_}",
                    f"bwd_skip={bench._async_backward_skip_reason or '-'}",
                    f"endpts_fwd={_s.get('n_fwd',0)}",
                    f"endpts_bwd={_s.get('n_bwd',0)}",
                    f"bwd_valid={_s.get('n_bwd_valid',0)}",
                    f"pairs={_s.get('n_pairs',0)}",
                    f"shadow={_s.get('shadow_rays',0)}",
                    f"visible={_s.get('visible',0)}",
                    flush=True,
                )

    def _rebuild_bench_with_iris(new_r_inner: float) -> None:
        """Full scene rebuild with a new iris r_inner (new f-number).

        This is the only correct path — the iris geometry is baked into the BVH at
        init time and cannot be patched in-place.  The rebuild takes ~1-2 s but
        gives a physically accurate aperture stop.
        """
        nonlocal bench
        # Clamp to reasonable range (1 mm – 30 mm clear aperture)
        new_r = float(np.clip(new_r_inner, 0.001, 0.030))
        iris = getattr(scene, "iris_aperture", None)
        if iris is None:
            print("[aperture] no iris_aperture on scene; cannot rebuild", flush=True)
            return
        import dataclasses as _dc
        scene.iris_aperture = _dc.replace(
            iris,
            r_inner=round(new_r, 5),
            r_outer=round(max(new_r * 1.55, float(iris.r_outer)), 5),
        )
        # Reset the optical design spec so the solver re-runs with the new geometry.
        scene.optical_design = _default_optical_design_spec()
        # Reset GL-side state that references old BVH data.
        _scene_vao[0]           = None
        _mesh_vbo[0]            = None
        _mesh_n_verts[0]        = 0
        _uv_last_update_s[0]    = -1.0
        _uv_shared_tex_id[0]    = 0
        print("[aperture] rebuilding scene…", flush=True)
        bench = ForwardCppLensBench(
            scene=scene,
            freq_hz=bench.freq_hz.copy(),
            view_h=view_h,
            view_w=view_w,
            sidecar=sidecar,
            field_capture=field_capture,
        )
        bench.sensor_amp_gain      = float(sensor_amp_gain)
        bench.sensor_min_amplitude = float(sensor_min_amplitude)
        bench.emitter_amp_gain     = float(emitter_amp_gain)
        bench.compute_mode         = str(compute_mode)
        if _gl_display_hglrc:
            if _gl_display_hdc:
                bench.tracer.set_gl_display_hdc(_gl_display_hdc)
            bench.tracer.set_gl_display_hglrc(_gl_display_hglrc)
            try:
                _wl_nm = np.clip(C_LIGHT / np.maximum(bench.freq_hz, EPS) * 1.0e9, 380.0, 700.0)
                _blit_w = _wavelength_to_rgb_weights(_wl_nm).astype(np.float32)
                bench.tracer.set_uv_blit_weights(_blit_w, mode=0)
            except Exception:
                pass
        _new_pip_res = int(max(16, scene.image_plate.sensor_res))
        bench.tracer.configure_sensor_image(
            float(scene.image_plate.x), float(scene.image_plate.radius),
            _new_pip_res, 0.008,
        )
        bench.tracer.set_bdpt_sweep_trigger(_new_pip_res * _new_pip_res)
        # Derive and print the new f-number from EFL and entrance pupil radius
        _efl = float(getattr(getattr(scene, "optical_design", None),
                              "effective_focal_length_m", 0.0) or 0.0)
        _ep_r = float(new_r)
        _asm_ap = getattr(getattr(bench, "_lens_assembly", None), "acceptance_params", lambda: None)()
        if _asm_ap is not None:
            _ep_r = float(_asm_ap.get("ep_r", new_r))
        _fnum = (_efl / (2.0 * _ep_r)) if _ep_r > 1e-6 and _efl > 1e-6 else float("nan")
        print(
            f"[aperture] r_inner={new_r*1e3:.2f}mm"
            f"  f/{_fnum:.1f}"
            f"  EFL={_efl*1e3:.1f}mm"
            f"  rebuilt tris={bench.n_tris}",
            flush=True,
        )
        _restore_display_gl_context()

    try:
        while True:
            # First frame: restore the intended f/5.6 startup aperture by rebuilding
            # the scene so the iris geometry is present in the BVH.
            if _startup_iris_pending[0] and getattr(scene, "iris_aperture", None) is not None:
                _startup_iris_pending[0] = False
                _od_s = getattr(scene, "optical_design", None)
                _efl_s = float(getattr(_od_s, "effective_focal_length_m", 0.0) or 0.0)
                if _efl_s > 1e-4:
                    _r_f22 = _efl_s / (2.0 * 5.6)
                    print(f"[startup] f/5.6 -> r={_r_f22*1e3:.2f}mm  EFL={_efl_s*1e3:.1f}mm", flush=True)
                    _rebuild_bench_with_iris(_r_f22)

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
                                projection_axis = 1
                                auto_wiggle = False
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
                            elif k == pygame.K_p:
                                show_impact_points = not show_impact_points
                                print(f"[impact-points] {1 if show_impact_points else 0}", flush=True)
                            elif k == pygame.K_LEFTBRACKET:
                                _mods = pygame.key.get_mods()
                                if _mods & pygame.KMOD_SHIFT:
                                    # Shift+[ → close aperture (higher f-number, less DoF)
                                    _iris_now = getattr(bench.scene, "iris_aperture", None)
                                    if _iris_now is not None:
                                        _new_r = float(_iris_now.r_inner) * 0.80
                                        _rebuild_bench_with_iris(_new_r)
                                else:
                                    bench.intent_shuffle = max(0.0, bench.intent_shuffle - 0.1)
                                    bench.tracer.set_intent_shuffle(float(bench.intent_shuffle))
                                    print(f"[shuffle] {bench.intent_shuffle:.2f}", flush=True)
                            elif k == pygame.K_RIGHTBRACKET:
                                _mods = pygame.key.get_mods()
                                if _mods & pygame.KMOD_SHIFT:
                                    # Shift+] → open aperture (lower f-number, more DoF)
                                    _iris_now = getattr(bench.scene, "iris_aperture", None)
                                    if _iris_now is not None:
                                        _new_r = float(_iris_now.r_inner) * 1.25
                                        _rebuild_bench_with_iris(_new_r)
                                else:
                                    bench.intent_shuffle = min(1.0, bench.intent_shuffle + 0.1)
                                    bench.tracer.set_intent_shuffle(float(bench.intent_shuffle))
                                    print(f"[shuffle] {bench.intent_shuffle:.2f}", flush=True)
                            # ── Live lens element selection ──────────────────────
                            # , / .  — cycle selected element backward / forward
                            elif k == pygame.K_COMMA:
                                _n = len(list(getattr(bench.scene, "lens_stack", None) or []))
                                if _n > 0:
                                    bench._mobile_lens_idx = (bench._mobile_lens_idx - 1) % _n
                                    print(f"[lens-select] element {bench._mobile_lens_idx}/{_n-1}", flush=True)
                            elif k == pygame.K_PERIOD:
                                _n = len(list(getattr(bench.scene, "lens_stack", None) or []))
                                if _n > 0:
                                    bench._mobile_lens_idx = (bench._mobile_lens_idx + 1) % _n
                                    print(f"[lens-select] element {bench._mobile_lens_idx}/{_n-1}", flush=True)
                            # ── Live lens element translation ────────────────────
                            # -  / =  — move element -/+ 0.1 mm  (×10 with Shift)
                            elif k == pygame.K_MINUS:
                                _mods = pygame.key.get_mods()
                                _step = -0.001 if (_mods & pygame.KMOD_SHIFT) else -0.0001
                                bench._adjust_lens_element(bench._mobile_lens_idx, _step)
                            elif k == pygame.K_EQUALS:
                                _mods = pygame.key.get_mods()
                                _step = 0.001 if (_mods & pygame.KMOD_SHIFT) else 0.0001
                                bench._adjust_lens_element(bench._mobile_lens_idx, _step)
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
                    for _ci in range(6):
                        _svc[_ci] = new_vbc[_ci]
                except Exception as exc:
                    if frame_profiler is not None:
                        frame_profiler.end("display_records")
                        frame_profiler.end("upload_volume")
                    print(f"[display] {exc}", flush=True)

            if projection_axis == 2:
                shader_mode = 2
                axis_label = "X/Z"
            else:
                shader_mode = 1
                axis_label = "X/Y"

            pygame.display.set_caption(
                f"Thick Lens — {axis_label} | bounces={max_bounces} | "
                f"shuffle={bench.intent_shuffle:.2f} ([/] to adjust)"
            )

            # Apply any pending LUT/MLP transition from the background worker.
            bench._poll_assembly_mode_switch()

            glClearColor(8/255, 8/255, 10/255, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glDepthFunc(GL_LEQUAL)
            glViewport(0, 0, W, H)
            t_now = pygame.time.get_ticks() * 0.001
            if frame_profiler is not None:
                frame_profiler.begin("draw_uv_mesh")
            draw_uv_mesh(volume_gain, 0.0)
            if show_impact_points:
                draw_surface_points(shader_mode, volume_gain, 0.0)
            draw_acceptance_cones(t_now, shader_mode)
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
        # 0. Stop progressive bake thread before tearing down the tracer.
        if bench._lens_assembly is not None:
            bench._lens_assembly.stop_progressive_refinement()
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
        glDeleteTextures(1, [tex_forward_pip])
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
        glDeleteProgram(cone_prog)
        glDeleteProgram(pip_prog)
        glDeleteProgram(mesh_prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        glDeleteShader(pvs)
        glDeleteShader(pfs)
        glDeleteShader(cvs)
        glDeleteShader(cfs)
        glDeleteShader(pip_vs)
        glDeleteShader(pip_fs)
        glDeleteShader(mvs)
        glDeleteShader(mfs)
        pygame.quit()


def run_uv_smoke(
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
    _batch_pip_res = int(max(16, scene.image_plate.sensor_res))
    bench.tracer.configure_sensor_image(
        float(scene.image_plate.x),
        float(scene.image_plate.radius),
        _batch_pip_res,
        0.008,
    )
    bench.tracer.set_bdpt_sweep_trigger(_batch_pip_res * _batch_pip_res)

    rc = 1
    try:
        print(
            "[uv-smoke-start]",
            f"compute={compute_mode}",
            f"sensor_res={scene.image_plate.sensor_res}",
            f"steps={int(max(1, steps))}",
            flush=True,
        )
        for step in range(int(max(1, steps))):
            seed = 20260516 + step * 101
            bench.trace_forward(1, seed, max_bounces=1)
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
    _ap.add_argument(
        "--bake-noodles",
        type=int,
        default=0,
        metavar="N",
        help="Number of rays per focus slice for neural camera training",
    )
    _ap.add_argument(
        "--bake-wavelengths",
        type=int,
        default=3,
        metavar="N",
        help="Wavelength samples averaged per baked noodle (default: 3)",
    )
    _ap.add_argument(
        "--bake-training-table",
        default="",
        metavar="PATH",
        help="Write a neural-lens training table .npy before tracing",
    )
    _ap.add_argument(
        "--bake-training-gb",
        type=float,
        default=0.0,
        metavar="GB",
        help="Size of neural-lens training table to stream in GiB; includes wavelength and focus columns",
    )
    _ap.add_argument(
        "--neural-assembly",
        action="store_true",
        help=(
            "Replace the full optical assembly with a trained MLP stand-in. "
            "Bakes per-band training data (4-dimensional: aperture, field, focus, wavelength), "
            "trains the network, exports a magic-14948 payload, and registers it as the "
            "optical context.  Use --bake-training-gb to control data size, "
            "--focus-steps / --focus-range-mm for the focus sweep, "
            "--neural-payload-out to save the payload, "
            "--neural-payload-in to skip baking and load a pre-trained payload."
        ),
    )
    _ap.add_argument(
        "--neural-train-epochs",
        type=int,
        default=60,
        metavar="N",
        help="Training epochs for --neural-assembly (default: 60)",
    )
    _ap.add_argument(
        "--neural-payload-in",
        default="",
        metavar="PATH",
        help="Load a pre-trained .npy neural-assembly payload and skip baking/training",
    )
    _ap.add_argument(
        "--neural-payload-out",
        default="",
        metavar="PATH",
        help="Save the trained neural-assembly payload to this .npy path",
    )
    _ap.add_argument(
        "--focus-steps",
        type=int,
        default=1,
        metavar="N",
        help="Number of focus positions to sweep for neural camera training",
    )
    _ap.add_argument(
        "--focus-range-mm",
        type=float,
        default=2.0,
        metavar="MM",
        help="Total sensor z sweep range for --focus-steps (default: 2.0 mm, centred on nominal)",
    )
    _ap.add_argument(
        "--parametric",
        action="store_true",
        help=(
            "Use exact algebraic CompoundLens transform instead of LUT or MLP. "
            "Evaluates the closed-form parametric equation of the full optical assembly "
            "per ray hit; no baking or training required."
        ),
    )
    _args = _ap.parse_args()
    if _args.uv_smoke_exit:
        raise SystemExit(run_uv_smoke(
            sensor_res=_args.sensor_res,
            sensor_amp_gain=_args.sensor_gain,
            sensor_min_amplitude=_args.sensor_min_amp,
            compute_mode=_args.compute_mode,
            steps=_args.uv_smoke_steps,
        ))
    run(
        sensor_res=_args.sensor_res,
        sensor_amp_gain=_args.sensor_gain,
        sensor_min_amplitude=_args.sensor_min_amp,
        emitter_amp_gain=_args.emitter_gain,
        compute_mode=_args.compute_mode,
        profile=bool(_args.profile),
        field_capture=not _args.no_field,
        bake_noodles=_args.bake_noodles,
        focus_steps=_args.focus_steps,
        focus_range_mm=_args.focus_range_mm,
        bake_wavelengths=_args.bake_wavelengths,
        bake_training_table=_args.bake_training_table,
        bake_training_gb=_args.bake_training_gb,
        neural_assembly=bool(_args.neural_assembly),
        neural_train_epochs=int(_args.neural_train_epochs),
        neural_payload_in=_args.neural_payload_in,
        neural_payload_out=_args.neural_payload_out,
        parametric=bool(_args.parametric),
    )
