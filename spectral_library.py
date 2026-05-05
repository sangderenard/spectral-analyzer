"""spectral_library
─────────────────────────────────────────────────────────────────────────────
Loads `configs/profiles/spectral_profiles.yaml` into the global
``EmissionProfileDatabase`` once at engine startup.  Every named profile
referenced by a material YAML must exist here — there is no inline-profile
authoring path in material YAMLs themselves.

Conventions
───────────
* The loader is **idempotent** at the registration level: re-loading the
  same library does not duplicate entries (names that already exist are
  overwritten in place by the underlying database).
* Names from the YAML are passed through unchanged (no prefixing).  A
  ``color:`` entry named ``ruby_body`` becomes the EPDB key
  ``ruby_body``.  A ``remission:`` entry named ``ruby_afterglow`` becomes
  the EPDB key ``ruby_afterglow``.  A material YAML references these by
  setting ``color_profile_name: ruby_body`` etc.
* Loading is eager: every profile's spectral→sRGB triple is baked into
  ``EmissionProfileDatabase._rgb_tensor`` before this function returns.
  No spectral resolution ever happens in the draw loop.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from material_db import (
    ColorProfile,
    EmissionProfile,
    EmissionProfileDatabase,
    HistogramSpread,
    ParametricSpread,
    RemissionProfile,
    RemissionResponseEnvelope,
    SpreadProfile,
)

_DEFAULT_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", "profiles", "spectral_profiles.yaml",
)


def _spread_term(d: dict) -> ParametricSpread:
    """Build a ParametricSpread from a YAML dict ``{amp, center, q}``."""
    return ParametricSpread(
        amp=float(d.get("amp",    1.0)),
        center=float(d.get("center", 550.0)),
        q=float(d.get("q",      1.0)),
    )


def _build_spread_terms(entry: Any) -> list:
    """Accept either a single spread dict or a list of spread dicts; return a
    list of ParametricSpread objects."""
    if entry is None:
        return []
    if isinstance(entry, dict):
        return [_spread_term(entry)]
    if isinstance(entry, list):
        return [_spread_term(d) for d in entry]
    raise TypeError(f"unexpected spread definition: {entry!r}")


def _build_spread_profile(spec: dict) -> SpreadProfile:
    """Build a SpreadProfile from a per-dimension dict.  Only `frequency` is
    populated for now (color and remission both express via spectral
    distribution alone); other dimensions default to neutral spreads."""
    sp = SpreadProfile()
    if "frequency" in spec:
        sp.frequency = _build_spread_terms(spec["frequency"])
    if "amplitude" in spec:
        sp.amplitude = _build_spread_terms(spec["amplitude"])
    if "phase" in spec:
        sp.phase = _build_spread_terms(spec["phase"])
    if "angular" in spec:
        sp.angular = _build_spread_terms(spec["angular"])
    return sp


def _build_envelope(spec: Optional[dict]) -> Optional[RemissionResponseEnvelope]:
    """Build a RemissionResponseEnvelope from a YAML dict, or return None.

    Recognised keys:
        decay_frames  int   length of the impulse response
        gain          float global scalar applied after sampling
        curve         (optional) Currently ignored — when authored, would
                      reference a parametric curve asset by name; for now,
                      omitting `curve` selects the default exponential decay.
    """
    if not spec:
        return None
    return RemissionResponseEnvelope(
        curve=None,                                      # asset linkage TBD
        decay_frames=int(spec.get("decay_frames", 8)),
        gain=float(spec.get("gain", 1.0)),
    )


def load_library(path: str = _DEFAULT_LIBRARY_PATH) -> EmissionProfileDatabase:
    """Load a spectral library YAML and register every entry into the global
    EmissionProfileDatabase.  Returns the (now-populated) database.

    All three top-level sections are optional:
        color:      {<name>: {frequency: [...], ...}, ...}    → ColorProfile
        emission:   {<name>: {spd: [...], total_power_W, ...}, ...} → EmissionProfile
        remission:  {<name>: {frequency: [...], response_envelope: {...}}, ...}
                                                              → RemissionProfile
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    epdb = EmissionProfileDatabase.instance()

    # ── Colour (reflectance) profiles ────────────────────────────────────────
    for name, spec in (data.get("color") or {}).items():
        cp = ColorProfile(spread=_build_spread_profile(spec or {}))
        epdb.register(name, cp)

    # ── Emission (light source) profiles ─────────────────────────────────────
    for name, spec in (data.get("emission") or {}).items():
        spec = spec or {}
        spd_terms = _build_spread_terms(spec.get("spd"))
        ep = EmissionProfile(
            spd=spd_terms if spd_terms else ParametricSpread(),
            peak_wavelength_nm=float(spec.get("peak_wavelength_nm", 550.0)),
            fwhm_nm=float(spec.get("fwhm_nm", 30.0)),
            total_power_W=float(spec.get("total_power_W", 1.0)),
        )
        epdb.register(name, ep)

    # ── Re-emission (frame-delayed) profiles ────────────────────────────────
    for name, spec in (data.get("remission") or {}).items():
        spec = spec or {}
        # Build a SpreadProfile carrying just the frequency field — the
        # remission's spectral character.  build_rgb_tensor() will integrate
        # this against the CIE CMFs for the colour channel.
        sp = SpreadProfile()
        if "frequency" in spec:
            sp.frequency = _build_spread_terms(spec["frequency"])
        rp = RemissionProfile(
            spread=sp,
            response_envelope=_build_envelope(spec.get("response_envelope")),
        )
        epdb.register(name, rp)

    return epdb


def load_default_library() -> EmissionProfileDatabase:
    """Convenience wrapper: load the canonical project library."""
    return load_library(_DEFAULT_LIBRARY_PATH)
