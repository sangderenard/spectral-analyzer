"""Validated participating-medium extension of the canonical material schema.

Definitions live only in configs/materials YAML and MaterialDatabase. This
module is a cold schema/units adapter, not a registry or a scattering solver.
Surface PBR values and debug sprite alpha are never optical coefficients.
"""
from __future__ import annotations
import hashlib
import json
import math

SCHEMA = "participating-medium-v1"
_PHASES = {"molecular_gas", "solid_aerosol", "liquid_aerosol"}


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def validate_medium_profile(value):
    """Return an independent validated profile; preserve explicit unknowns."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("medium must declare participating-medium-v1")
    try:
        profile = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("medium profile must be a finite JSON-compatible tree") from error
    if profile.get("phase") not in _PHASES:
        raise ValueError("medium phase must distinguish gas, solid aerosol and liquid aerosol")
    if not isinstance(profile.get("formula"), str) or not profile["formula"].strip():
        raise ValueError("medium requires a chemical formula or identified mixture")
    for key in ("molar_mass_kg_mol", "particle_material_density_kg_m3"):
        if profile.get(key) is not None:
            _number(profile[key], key, positive=True)
    if profile["phase"] == "molecular_gas" and profile.get("particle_material_density_kg_m3") is not None:
        raise ValueError("molecular gas cannot use a condensed particle material density")
    provenance = profile.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("medium profile requires source provenance")
    documented = set()
    for entry in provenance:
        if not isinstance(entry, dict) or not isinstance(entry.get("property"), str):
            raise ValueError("medium provenance must identify a property")
        if not isinstance(entry.get("url"), str) or not entry["url"].startswith("https://"):
            raise ValueError("medium provenance requires an HTTPS source URL")
        if not isinstance(entry.get("note"), str) or not entry["note"].strip():
            raise ValueError("medium provenance must distinguish source facts and model assumptions")
        documented.add(entry["property"])
    for key in ("molar_mass_kg_mol", "particle_material_density_kg_m3"):
        if profile.get(key) is not None and key not in documented:
            raise ValueError(f"medium {key} requires property-specific provenance")
    optics = profile.get("optics")
    if not isinstance(optics, dict):
        raise ValueError("medium requires an explicit optics mapping")
    model = optics.get("model")
    if model not in ("unparameterized", "geometric_optics_spheres"):
        raise ValueError("unsupported participating-medium optical model")
    # These properties remain unknown unless explicitly authored. Extinction
    # does not determine the scattering/absorption split or angular phase law.
    for key in ("mass_absorption_m2_kg", "mass_scattering_m2_kg"):
        if optics.get(key) is not None:
            _number(optics[key], "optics." + key)
    if optics.get("single_scattering_albedo") is not None:
        if not 0 <= _number(optics["single_scattering_albedo"], "single_scattering_albedo") <= 1:
            raise ValueError("single scattering albedo must lie in [0, 1]")
    if optics.get("asymmetry_g") is not None:
        g = optics["asymmetry_g"]
        if type(g) not in (int, float) or not math.isfinite(g) or not -1 <= g <= 1:
            raise ValueError("asymmetry_g must lie in [-1, 1]")
    if model == "unparameterized":
        if optics.get("mass_extinction_m2_kg") is not None:
            raise ValueError("unparameterized optical data cannot assert an extinction coefficient")
        if not isinstance(optics.get("required_inputs"), list) or not optics["required_inputs"]:
            raise ValueError("unparameterized optics must state the missing inputs")
    else:
        if profile["phase"] == "molecular_gas":
            raise ValueError("geometric particle optics cannot model molecular vapor")
        _number(profile.get("particle_material_density_kg_m3"), "particle_material_density_kg_m3", positive=True)
        _number(optics.get("particle_radius_m"), "particle_radius_m", positive=True)
        if type(optics.get("extinction_efficiency")) not in (int, float) or optics["extinction_efficiency"] != 2.0:
            raise ValueError("geometric optics sphere limit uses Q_ext = 2, not a fitted gain")
        if optics.get("size_distribution") != "monodisperse_reference":
            raise ValueError("sphere reference must declare its monodisperse assumption")
        interval = optics.get("wavelength_interval_m")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("geometric optics reference requires a wavelength interval in metres")
        lo, hi = (_number(v, "wavelength_interval_m", positive=True) for v in interval)
        if hi <= lo:
            raise ValueError("wavelength interval must increase")
        if not isinstance(optics.get("approximation"), str) or not optics["approximation"].strip():
            raise ValueError("geometric optics reference must disclose its approximation")
        if "optics.model" not in documented or "optics.particle_radius_m" not in documented:
            raise ValueError("particle optics needs model provenance and an explicit size assumption")
        if optics.get("mass_extinction_m2_kg") is not None:
            raise ValueError("sphere extinction is derived from Q, radius and density, not a second coefficient")
    return profile


def medium_profile_digest(profile):
    profile = validate_medium_profile(profile)
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mass_extinction_from_medium(profile, *, wavelengths_m, allow_approximation=False):
    """Resolve the declared cold coefficient for the existing scalar grid.

    Unknown gas spectra fail explicitly. The sole implemented particle model
    is an opt-in achromatic geometric-optics reference, not measured Mie data.
    All transport wavelengths must be supplied and inside its authored scope.
    """
    profile = validate_medium_profile(profile)
    optics = profile["optics"]
    if optics["model"] == "unparameterized":
        raise ValueError("medium optical data unavailable: " + "; ".join(map(str, optics["required_inputs"])))
    if allow_approximation is not True:
        raise ValueError("geometric optics reference requires explicit allow_approximation=True")
    if not isinstance(wavelengths_m, (list, tuple)) or not wavelengths_m:
        raise ValueError("declare all transport wavelengths in metres")
    low, high = optics["wavelength_interval_m"]
    for wavelength in wavelengths_m:
        wavelength = _number(wavelength, "wavelength_m", positive=True)
        if not low <= wavelength <= high:
            raise ValueError("transport wavelength is outside the declared particle model interval")
    radius = optics["particle_radius_m"]
    density = profile["particle_material_density_kg_m3"]
    coefficient = 3.0 * optics["extinction_efficiency"] / (4.0 * density * radius)
    if not math.isfinite(coefficient):
        raise ValueError("derived mass extinction is not finite")
    return coefficient
