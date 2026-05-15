"""Photographic exposure budget for ray-traced calibration.

Goal
----
The calibration loop needs a *known system of equations* relating

    (camera + film + lens settings, exposure time, scene radiance)
        →  total ray energy that should arrive at the sensor

so the gain search has zero ambiguity.  This module gives that bridge.

Once the target sensor-plane radiant exposure is known, we know how many
rays at what total energy must be emitted from emissive surfaces, and we
spread that ray budget across batches/frames so the running sum equals
the target — exactly the way a real shutter accumulates photons over its
open interval.

The module is intentionally renderer-agnostic.  It does NOT touch the
SensorAccumulator or any GL state; any harness that has a Camera (focal
length + sensor pitch + aperture + iso + exposure time) and a way to
emit batches of rays can use it.

Equations
---------
Let:

    f       = focal length            [m]    (focal_mm * 1e-3)
    D       = entrance pupil diameter [m]
    N       = f-number  = f / D
    p       = pixel pitch             [m]
    t       = exposure (shutter) time [s]
    L       = scene radiance          [W·sr⁻¹·m⁻²]
    Ω       = solid angle subtended by the lens at a sensor pixel

    Ω      ≈ π · sin²θ_max   with sin θ_max = 1 / sqrt(1 + 4 N²)
    Φ_pix  = L · A_pix · Ω · τ_lens                  [W]    per pixel
    H_pix  = Φ_pix · t                               [J]    per pixel
    ISO scaling:  film/sensor sensitivity multiplies H_pix → image gain.

    N_photons_pix(λ) = H_pix / (h · c / λ)

For an isotropic Lambertian emitter of total power P_emit_W, the radiance
seen at the lens (after free-space falloff, neglecting occlusion) is
    L_at_lens = P_emit_W / (π · A_emit_m²)
provided the emitter fills the field of view.  Use this only for sanity
checks against the per-emitter ``total_power_W`` carried by EmissionProfile.

Ray budget
----------
If we model each ray as carrying energy ``E_ray`` and the sensor must
collect total radiant exposure ``H_total = sum_pix(H_pix)``, then::

    N_rays_total =  H_total / (E_ray · η_capture)

where η_capture is the fraction of emitted rays that actually reach the
sensor (camera-back hit rate).  We don't know η_capture analytically,
but the bidirectional solve discovers it: the calibration loop runs the
forward emission + backward camera trace, measures how many rays land,
and adjusts gain so that the integrated sensor signal equals H_total.

This module gives the budget allocator that splits N_rays_total across
batches/frames so the camera "shutter" never exceeds its open interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

H_PLANCK   = 6.62607015e-34         # J·s
C_LIGHT    = 2.99792458e8           # m/s
PI         = math.pi


# ── Core dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CameraOptics:
    """Minimum information needed to compute pixel-plane irradiance.

    All units SI.  ``f_number`` is computed from focal_mm / aperture_mm if
    aperture_mm > 0, otherwise it falls back to ``max_aperture_fstop``.
    """
    focal_mm:           float = 35.0
    aperture_mm:        float = 0.0       # entrance pupil diameter (0 → use max f/N)
    max_aperture_fstop: float = 1.4
    pixel_pitch_um:     float = 8.4
    sensor_w_mm:        float = 36.0
    sensor_h_mm:        float = 24.0
    lens_transmission:  float = 0.95      # average τ across visible band

    @property
    def f_number(self) -> float:
        if self.aperture_mm > 1.0e-6:
            return float(self.focal_mm / self.aperture_mm)
        return float(self.max_aperture_fstop)

    @property
    def pixel_area_m2(self) -> float:
        p = self.pixel_pitch_um * 1.0e-6
        return p * p

    def lens_solid_angle_sr(self) -> float:
        """Solid angle subtended by the entrance pupil at a sensor pixel.

        Uses the exact ``Ω = π sin²θ_max`` for a circular cone with
        ``sin θ_max = 1 / sqrt(1 + 4 N²)`` (paraxial-free; no small-angle).
        """
        N = max(self.f_number, 0.5)
        sin_t = 1.0 / math.sqrt(1.0 + 4.0 * N * N)
        return PI * sin_t * sin_t

    def n_pixels(self) -> int:
        nx = max(1, round(self.sensor_w_mm * 1.0e-3 /
                          (self.pixel_pitch_um * 1.0e-6)))
        ny = max(1, round(self.sensor_h_mm * 1.0e-3 /
                          (self.pixel_pitch_um * 1.0e-6)))
        return int(nx * ny)


@dataclass
class FilmExposure:
    """Film/sensor side of the exposure equation."""
    iso:               float = 100.0     # ISO 5800 sensitivity
    exposure_time_s:   float = 1.0 / 60  # shutter open interval
    quantum_efficiency: float = 0.5      # photons → e⁻ conversion (0..1)
    target_mid_grey:   float = 0.18      # 18% reflectance grey target


@dataclass
class RayDispatchPlan:
    """How to spread the ray budget over batches / frames so the running
    sum reaches the exposure target exactly when the shutter closes."""
    total_rays:        int           # rays to emit during full exposure
    n_batches:         int           # batches to split the budget into
    rays_per_batch:    int           # ceil(total_rays / n_batches)
    energy_per_ray_J:  float         # nominal energy carried per ray
    target_H_J:        float         # total radiant exposure on the sensor
    target_photons:    float         # total photons across all pixels at λ_ref
    ref_wavelength_nm: float         # λ used to convert J ↔ photon count


# ── Equations ────────────────────────────────────────────────────────────────

def pixel_irradiance_W_per_m2(L_radiance_W_sr_m2: float,
                              optics: CameraOptics) -> float:
    """Irradiance at the sensor plane for a uniform scene radiance ``L``.

        E_pix = L · Ω_lens · τ_lens         [W·m⁻²]
    """
    return float(L_radiance_W_sr_m2 *
                 optics.lens_solid_angle_sr() *
                 optics.lens_transmission)


def pixel_radiant_exposure_J(L_radiance_W_sr_m2: float,
                             optics: CameraOptics,
                             film: FilmExposure) -> float:
    """Radiant exposure ``H = Φ · t`` per pixel (Joules)."""
    E_W_m2 = pixel_irradiance_W_per_m2(L_radiance_W_sr_m2, optics)
    Phi_W   = E_W_m2 * optics.pixel_area_m2
    return float(Phi_W * film.exposure_time_s)


def photons_per_pixel(L_radiance_W_sr_m2: float,
                      optics: CameraOptics,
                      film: FilmExposure,
                      wavelength_nm: float = 555.0) -> float:
    """Photon count per pixel (uses ``λ`` to set photon energy)."""
    H_J = pixel_radiant_exposure_J(L_radiance_W_sr_m2, optics, film)
    E_photon_J = H_PLANCK * C_LIGHT / max(wavelength_nm * 1.0e-9, 1.0e-12)
    return float(H_J / E_photon_J)


def lambertian_emitter_radiance(total_power_W: float,
                                area_m2: float) -> float:
    """Radiance of a Lambertian emitter::

        L = M / π = (P / A) / π   [W·sr⁻¹·m⁻²]
    """
    if area_m2 <= 0.0:
        return 0.0
    return float(total_power_W / (PI * area_m2))


# ── Budget allocator ─────────────────────────────────────────────────────────

def plan_ray_budget(
    optics: CameraOptics,
    film: FilmExposure,
    *,
    scene_radiance_W_sr_m2: float,
    rays_per_pixel_per_second: float = 1.0,
    n_batches: Optional[int] = None,
    sensor_fps: float = 0.0,
    sensor_spp: float = 1.0,
    capture_efficiency: float = 1.0,
    ref_wavelength_nm: float = 555.0,
) -> RayDispatchPlan:
    """Compute a ray dispatch plan for the given exposure.

    Parameters
    ----------
    optics, film
        Camera/film configuration.
    scene_radiance_W_sr_m2
        Scene-side radiance the calibration is hitting (e.g. from
        ``lambertian_emitter_radiance(profile.total_power_W, area)``).
    rays_per_pixel_per_second
        Author-side density knob.  ``ray_density`` × ``sensor_spp`` from
        the harness controls maps onto this.  Total rays scales linearly
        with this value.
    n_batches
        How many separate dispatches the shutter interval should be split
        into.  When ``None`` the value is derived from ``sensor_fps`` and
        ``film.exposure_time_s``: max(1, round(fps * exposure_time)).
    sensor_fps, sensor_spp
        Real frame rate and per-pixel sample multiplier.  ``sensor_fps``
        is used only to derive ``n_batches`` when not supplied.
    capture_efficiency
        Fraction of emitted rays that reach the sensor in the
        bidirectional solve.  ``1.0`` ≡ "all rays count" (upper bound).
        The calibration loop discovers the true value and divides through
        before the next plan.
    ref_wavelength_nm
        Reference wavelength for converting Joules ↔ photon count.

    Returns
    -------
    RayDispatchPlan
    """
    # ── Per-pixel exposure and total scene exposure target ────────────────
    H_pix_J  = pixel_radiant_exposure_J(scene_radiance_W_sr_m2, optics, film)
    n_pix    = optics.n_pixels()
    H_tot_J  = H_pix_J * n_pix

    E_photon_J     = H_PLANCK * C_LIGHT / max(ref_wavelength_nm * 1.0e-9, 1.0e-12)
    total_photons  = H_tot_J / E_photon_J

    # ── Total rays demanded by the author-side density knob ───────────────
    spp = max(0.0, float(sensor_spp))
    # Allow sub-1 ray/pixel budgets so callers can request low total-ray
    # quick previews (e.g., 100k rays on a multi-megapixel sensor).
    rays_per_pix = max(0.0, float(rays_per_pixel_per_second) * spp)
    total_rays = int(round(rays_per_pix * n_pix))
    total_rays = max(total_rays, 1)

    # ── Energy per ray so that Σ rays × E_ray = target H (capture-aware) ──
    eta = max(1.0e-6, float(capture_efficiency))
    E_ray_J = float(H_tot_J) / float(total_rays * eta)

    # ── Split across batches over the shutter interval ────────────────────
    if n_batches is None:
        if sensor_fps > 0.0 and film.exposure_time_s > 0.0:
            n_batches = max(1, int(round(sensor_fps * film.exposure_time_s)))
        else:
            n_batches = 1
    n_batches = max(1, int(n_batches))
    rays_per_batch = max(1, (total_rays + n_batches - 1) // n_batches)

    return RayDispatchPlan(
        total_rays        = int(total_rays),
        n_batches         = int(n_batches),
        rays_per_batch    = int(rays_per_batch),
        energy_per_ray_J  = float(E_ray_J),
        target_H_J        = float(H_tot_J),
        target_photons    = float(total_photons),
        ref_wavelength_nm = float(ref_wavelength_nm),
    )


# ── Notes on the bidirectional solve ─────────────────────────────────────────
#
# This module DOES NOT implement a bidirectional path tracer.  The existing
# BDPT pipeline lives in ``demo_pluck_gl.py``:
#
#   forward pass   : ``_GPU_RAY_FIELD_CS`` (PASS_FORWARD), driven by
#                    ``SensorAccumulator.pump_forward()`` over the
#                    ``bdpt_sources`` SSBO baked from
#                    ``RayOrder.bake_rays`` / ``pack_emissive_area_rays``.
#   backward pass  : ``_GPU_SENSOR_CS`` (PASS_BACKWARD), driven by
#                    ``SensorAccumulator.tick()`` when sensor layer 9 is
#                    visible (see ``Renderer.tick_sensor()``).
#
# Calibration callers should:
#   1. Call ``plan_ray_budget(...)`` to derive ``rays_per_batch`` and
#      the per-batch ``target_H_J`` slice for the current camera+film state.
#   2. Each batch, call
#        ``pack_emissive_area_rays(..., n_rays=plan.rays_per_batch,
#                                  total_energy_J=plan.target_H_J / plan.n_batches)``
#      and upload the rows to ``bdpt_sources``.
#   3. Let ``Renderer.tick_sensor()`` run the existing forward+backward BDPT.
#   4. Read the integrated sensor texture and adjust the gain so the
#      measured radiant exposure matches ``plan.target_H_J``.
#
# Anything beyond that mapping is implemented by the existing BDPT, NOT here.


def summarize_plan(plan: RayDispatchPlan, optics: CameraOptics,
                   film: FilmExposure) -> dict:
    """Render a plan to a flat dict suitable for log-line printing."""
    return {
        "f_number":             round(optics.f_number, 3),
        "pixel_pitch_um":       optics.pixel_pitch_um,
        "n_pixels":             optics.n_pixels(),
        "lens_omega_sr":        round(optics.lens_solid_angle_sr(), 6),
        "exposure_time_s":      film.exposure_time_s,
        "iso":                  film.iso,
        "target_H_J":           plan.target_H_J,
        "target_photons":       plan.target_photons,
        "total_rays":           plan.total_rays,
        "n_batches":            plan.n_batches,
        "rays_per_batch":       plan.rays_per_batch,
        "energy_per_ray_J":     plan.energy_per_ray_J,
        "ref_wavelength_nm":    plan.ref_wavelength_nm,
    }
