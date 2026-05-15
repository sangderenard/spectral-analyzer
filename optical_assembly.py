"""Optical assembly and element data model for tier-based camera transport.

This module defines the data structures that separate visual mesh geometry from
optical behavior, enabling explicit representation of optical elements (stops,
mirrors, refractors, wave regions) across the 7-tier camera transport hierarchy.

See CAMERA_SYSTEM_REFACTORING.md Phase B for full architectural specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optical element roles (handlers in C++ ray tracer)
# ─────────────────────────────────────────────────────────────────────────────

class OpticalElementRole(str, Enum):
    """Optical behavior handler for a surface element."""
    STOP = "stop"                          # Aperture stop (blocks rays outside diameter)
    MIRROR = "mirror"                      # Specular reflection
    REFRACTOR = "refractor"                # Refraction through glass/dielectric
    FILTER = "filter"                      # Wavelength-dependent absorption
    DIFFUSER = "diffuser"                  # Lambertian scattering
    SENSOR = "sensor"                      # Terminal surface (photon collection)
    WAVE_REGION = "wave_region"            # Diffraction/interference zone

    def is_terminal(self) -> bool:
        """Returns True if this element is a final photon sink."""
        return self in (OpticalElementRole.SENSOR,)

    def is_stopping_surface(self) -> bool:
        """Returns True if this element can block ray propagation."""
        return self in (OpticalElementRole.STOP, OpticalElementRole.FILTER)

    def requires_material(self) -> bool:
        """Returns True if optical material properties are required."""
        return self in (
            OpticalElementRole.MIRROR,
            OpticalElementRole.REFRACTOR,
            OpticalElementRole.FILTER,
            OpticalElementRole.DIFFUSER,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Geometry specifications
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundarySpec:
    """Specification for a boundary surface (entrance, exit, or aperture)."""
    name: str
    # Shape: "plane", "sphere", "cylinder", "aspheric", "custom"
    shape: str = "plane"
    # Center position in optical bench coordinates (m)
    center_z_m: float = 0.0
    # Normal direction (always along optical axis for standard mounting)
    normal: str = "forward"  # "forward" or "backward"
    # Diameter or aperture extent (m)
    diameter_m: Optional[float] = None
    # Curvature radius for spherical/aspheric (m); positive = convex toward incident light
    radius_m: Optional[float] = None
    # Conic constant for aspheric (k = 0 for sphere)
    conic_k: float = 0.0


@dataclass
class GeometrySpec:
    """Specification for optical element geometry in CAD/optical design space."""
    # Unique element identifier (e.g. "stop_iris_7mm", "obj_lens_19mm")
    name: str
    # Position along optical bench z-axis (m)
    center_z_m: float = 0.0
    # Thickness for volumetric elements (m)
    thickness_m: float = 0.0
    # Diameter or major dimension (m)
    diameter_m: float = 0.0
    # For aspheric: conic constant k and polynomial coeffs
    conic_k: float = 0.0
    # Polynomial aspheric coefficients [a2, a4, a6, a8, ...]
    aspheric_coeffs: Optional[np.ndarray] = None
    # Visual mesh reference (from CAD/STEP model)
    mesh_name: Optional[str] = None


@dataclass
class OpticalMaterialSpec:
    """Specification for optical material properties."""
    # Material identifier (e.g. "BK7", "Fused Silica", "Al_mirror")
    name: str
    # Refractive index (real part; imag part for absorption handled separately)
    refractive_index: float = 1.5
    # Dispersion curve (wavelength [um] -> n [dimensionless])
    # Can be Cauchy, Sellmeier, or tabulated curve
    dispersion_model: str = "constant"  # "constant", "cauchy", "sellmeier", "table"
    # Dispersion coefficients (depends on model)
    dispersion_coeffs: Optional[dict[str, float]] = None
    # Absorption coefficient [cm^-1] at reference wavelength
    absorption_coeff_per_cm: float = 0.0
    # Reference wavelength for absorption (um)
    absorption_wavelength_um: float = 0.55
    # Thermal effects (dn/dT in 1/K)
    dn_dT_per_kelvin: float = 0.0
    # Surface roughness RMS (um)
    surface_roughness_um: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Optical element definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpticalElement:
    """A single optical element (lens, mirror, stop, filter, etc.)."""
    # Unique identifier within the assembly
    id: str
    # Optical behavior role (determines C++ handler)
    role: OpticalElementRole
    # Geometric specification
    geometry: GeometrySpec
    # Material specification (None for air/vacuum gaps or stops)
    material: Optional[OpticalMaterialSpec] = None
    # Name of handler in C++ ray tracer (e.g. "refractive_thick_lens")
    handler_name: str = ""
    # Whether this element participates in ray transport
    active: bool = True
    # Whether this element has a visual mesh (for display, not ray-trace)
    visual_only: bool = False
    # Reference to visual mesh group in scene (e.g. "obj_lens_1")
    visual_mesh_ref: Optional[str] = None
    # Custom telemetry tag for grouping (e.g. "entrance_optics", "field_optics")
    telemetry_group: str = "optical"

    def validate(self) -> Optional[str]:
        """Check consistency of element specification.

        Returns error message if invalid, None if OK.
        """
        if not self.id.strip():
            return "OpticalElement.id cannot be empty"

        if self.role.requires_material() and self.material is None:
            return f"OpticalElement {self.id} role={self.role.value} requires material specification"

        if self.role == OpticalElementRole.STOP and self.geometry.diameter_m <= 0:
            return f"OpticalElement {self.id} role=stop requires positive diameter_m"

        if self.geometry.diameter_m < 0:
            return f"OpticalElement {self.id} has negative diameter_m={self.geometry.diameter_m}"

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Optical assembly (full camera optical path)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpticalAssembly:
    """Complete optical transport assembly for a camera system.

    Separates visual mesh from optical behavior, defines the sequence of optical
    elements from entrance to sensor, and provides validation and constraints for
    each tier mode.
    """
    # Assembly identifier
    name: str
    # Optical bench reference coordinate system (z = optical axis)
    elements: list[OpticalElement] = field(default_factory=list)
    # Entrance boundary specification (where light enters the optical system)
    entrance_boundary: Optional[BoundarySpec] = None
    # Exit boundary specification (where light exits to sensor)
    exit_boundary: Optional[BoundarySpec] = None
    # Sensor specification (terminal collection surface)
    sensor: Optional[OpticalElement] = None
    # Mechanical stops/apertures (sublist of elements for quick lookup)
    stops: list[OpticalElement] = field(default_factory=list)
    # Visual mesh groups (for debugging/preview; not ray-traced)
    debug_visual_meshes: list[dict[str, str]] = field(default_factory=list)
    # Optical bench total length (m), for coordinate system bounds checking
    length_m: float = 0.1
    # Sensor plane z-position in assembly coordinates (m)
    sensor_z_m: float = 0.0
    # Free-space distance from entrance to first optical element (m)
    entrance_to_first_m: float = 0.0

    def validate(self) -> Optional[str]:
        """Comprehensive validation of optical assembly specification.

        Returns error message if invalid, None if OK.
        """
        if not self.name.strip():
            return "OpticalAssembly.name cannot be empty"

        # Validate all elements
        for elem in self.elements:
            elem_err = elem.validate()
            if elem_err is not None:
                return elem_err

        # Check that all stops are in elements list
        stop_ids = set(s.id for s in self.stops)
        elem_ids = set(e.id for e in self.elements)
        missing_stops = stop_ids - elem_ids
        if missing_stops:
            return f"OpticalAssembly stops reference undefined elements: {missing_stops}"

        # Check that sensor (if defined) is in elements
        if self.sensor is not None:
            if self.sensor.id not in elem_ids:
                return f"OpticalAssembly.sensor references undefined element {self.sensor.id}"
            if self.sensor.role != OpticalElementRole.SENSOR:
                return f"OpticalAssembly.sensor has role={self.sensor.role.value}, expected SENSOR"

        # Check z-ordering for physical consistency
        z_coords = sorted((e.geometry.center_z_m, e.id) for e in self.elements)
        if len(z_coords) > 1:
            for i in range(len(z_coords) - 1):
                z1, id1 = z_coords[i]
                z2, id2 = z_coords[i + 1]
                if z1 == z2:
                    return f"OpticalAssembly elements {id1} and {id2} have identical z={z1:.6f}m"

        # Verify bounds
        if self.length_m <= 0:
            return f"OpticalAssembly.length_m must be positive, got {self.length_m}"

        if self.entrance_to_first_m < 0:
            return f"OpticalAssembly.entrance_to_first_m cannot be negative"

        return None

    def active_elements(self) -> list[OpticalElement]:
        """Return list of active optical elements (excluding disabled ones)."""
        return [e for e in self.elements if e.active]

    def optical_surfaces(self) -> list[OpticalElement]:
        """Return list of elements with refractive/reflective surfaces."""
        return [
            e for e in self.elements
            if e.active and e.role in (
                OpticalElementRole.MIRROR,
                OpticalElementRole.REFRACTOR,
                OpticalElementRole.FILTER,
                OpticalElementRole.DIFFUSER,
            )
        ]

    def stopping_surfaces(self) -> list[OpticalElement]:
        """Return list of elements that can block ray propagation."""
        return [
            e for e in self.elements
            if e.active and e.role.is_stopping_surface()
        ]

    def has_diffractive_elements(self) -> bool:
        """Returns True if assembly includes diffraction zones."""
        return any(e.active and e.role == OpticalElementRole.WAVE_REGION for e in self.elements)

    def has_reflective_elements(self) -> bool:
        """Returns True if assembly includes mirrors."""
        return any(e.active and e.role == OpticalElementRole.MIRROR for e in self.elements)

    def to_dict(self) -> dict:
        """Export assembly to JSON-serializable dictionary."""
        return {
            "name": str(self.name),
            "length_m": float(self.length_m),
            "sensor_z_m": float(self.sensor_z_m),
            "entrance_to_first_m": float(self.entrance_to_first_m),
            "n_elements": len(self.elements),
            "n_active_elements": len(self.active_elements()),
            "n_stops": len(self.stops),
            "n_optical_surfaces": len(self.optical_surfaces()),
            "has_diffractive": bool(self.has_diffractive_elements()),
            "has_reflective": bool(self.has_reflective_elements()),
            "sensor_name": self.sensor.id if self.sensor is not None else None,
            "entrance_boundary": {
                "name": self.entrance_boundary.name,
                "shape": self.entrance_boundary.shape,
            } if self.entrance_boundary else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Assembly construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def assembly_from_simple_thin_lens(
    name: str,
    focal_length_m: float,
    aperture_diameter_m: float,
    sensor_distance_m: float,
) -> OpticalAssembly:
    """Quick constructor: simple thin lens + aperture stop + sensor.

    Used for bootstrapping simple camera modes (PINHOLE, APERTURE_CONE, THIN_LENS_GEOMETRIC).
    """
    lens_elem = OpticalElement(
        id="thin_lens_primary",
        role=OpticalElementRole.REFRACTOR,
        geometry=GeometrySpec(
            name="thin_lens",
            center_z_m=0.0,
            diameter_m=aperture_diameter_m,
        ),
        material=OpticalMaterialSpec(name="glass", refractive_index=1.5),
        handler_name="thin_lens",
        visual_mesh_ref="lens_cad",
    )

    stop_elem = OpticalElement(
        id="aperture_stop",
        role=OpticalElementRole.STOP,
        geometry=GeometrySpec(
            name="aperture_stop",
            center_z_m=0.0,
            diameter_m=aperture_diameter_m,
        ),
        handler_name="aperture_stop",
    )

    sensor_elem = OpticalElement(
        id="sensor_plane",
        role=OpticalElementRole.SENSOR,
        geometry=GeometrySpec(
            name="sensor",
            center_z_m=sensor_distance_m,
        ),
        handler_name="sensor",
        visual_only=True,
    )

    assembly = OpticalAssembly(
        name=name,
        elements=[lens_elem, stop_elem, sensor_elem],
        stops=[stop_elem],
        sensor=sensor_elem,
        sensor_z_m=sensor_distance_m,
        length_m=sensor_distance_m + 0.01,
    )

    return assembly


def assembly_from_solved_camera_package(
    solved_package: Optional[object],  # camera_parametric_solver.SolvedCameraPackage
    name: str = "solved_camera_assembly",
) -> OpticalAssembly:
    """Construct OpticalAssembly from a SolvedCameraPackage (parametric solver output).

    Maps the solved camera parameters (focal length, focus distance, aperture radius,
    lens position) into an OpticalAssembly suitable for tier 3+ (THIN_LENS_GEOMETRIC
    and above).

    Args:
        solved_package: Output from solve_sane_camera_rig()
        name: Assembly identifier

    Returns:
        OpticalAssembly or None if package is None or invalid
    """
    if solved_package is None:
        return None

    try:
        _si = solved_package.sanity_input
        _rpt = solved_package.sanity_report

        _eff_f = float(_rpt.planes.effective_focal_m)
        _fp_z = float(_rpt.planes.focal_plane_z_m)
        _ap_z = float(
            _si.aperture_rail_z_m if _si.aperture_rail_z_m is not None
            else _si.aperture_plane_offset_m
        )
        _s_z = float(_si.sensor_plane_z_m)
        _ap_radius = float(_si.aperture_radius_m or 0.005)
        _foc_dist = max(1.0e-4, abs(_fp_z - _ap_z))

        # Primary lens element (thin lens at aperture position)
        lens_elem = OpticalElement(
            id="thin_lens_from_solver",
            role=OpticalElementRole.REFRACTOR,
            geometry=GeometrySpec(
                name="solved_lens",
                center_z_m=_ap_z,
                diameter_m=2.0 * _ap_radius,
            ),
            material=OpticalMaterialSpec(
                name="optical_glass",
                refractive_index=1.5,
                dispersion_model="constant",
            ),
            handler_name="thin_lens",
        )

        # Aperture stop at lens position
        stop_elem = OpticalElement(
            id="solved_stop",
            role=OpticalElementRole.STOP,
            geometry=GeometrySpec(
                name="solved_stop",
                center_z_m=_ap_z,
                diameter_m=2.0 * _ap_radius,
            ),
            handler_name="aperture_stop",
        )

        # Sensor at sensor plane position
        sensor_elem = OpticalElement(
            id="sensor_plane",
            role=OpticalElementRole.SENSOR,
            geometry=GeometrySpec(
                name="sensor",
                center_z_m=_s_z,
            ),
            handler_name="sensor",
            visual_only=True,
        )

        # Calculate assembly bounds
        z_min = min(_ap_z, _s_z)
        z_max = max(_ap_z, _s_z)
        assembly_length = z_max - z_min + 0.01

        assembly = OpticalAssembly(
            name=name,
            elements=[lens_elem, stop_elem, sensor_elem],
            stops=[stop_elem],
            sensor=sensor_elem,
            sensor_z_m=_s_z,
            length_m=assembly_length,
            entrance_to_first_m=_ap_z,
        )

        return assembly

    except (AttributeError, TypeError) as e:
        print(f"  [warn] assembly_from_solved_camera_package failed: {e}")
        return None
