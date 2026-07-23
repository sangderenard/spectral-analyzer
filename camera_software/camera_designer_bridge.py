"""Canonical optical-engine camera views for the interactive GL designer.

The designer historically owned a separate ``CameraPreset`` catalogue.  This
module makes that preset a derived view of the production ``CameraManifest``:
the manifest remains authoritative, the ordinary four-group solver chooses the
physical positions, and the designer receives the same radii, thicknesses,
glass indices, aperture and 6x6 recording gate for drawing and fast scanning.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .camera_manifest import CameraManifest, resolve_camera_manifest
from .optical_design import OpticalDesignSpec, solve_four_group_zoom_surrogate


CONIC_TYPE_DEFAULT_K = {
    "sphere": 0.0,
    "paraboloid": -1.0,
    "hyperboloid": -1.5,
    "prolate_ellipsoid": -0.5,
    "oblate_ellipsoid": 0.5,
}


def conic_type_from_k(conic_k: float) -> str:
    """Classify a rotational conic from its conventional conic constant."""

    value = float(conic_k)
    if not math.isfinite(value):
        raise ValueError("conic_k must be finite")
    if math.isclose(value, 0.0, abs_tol=1.0e-12):
        return "sphere"
    if math.isclose(value, -1.0, abs_tol=1.0e-12):
        return "paraboloid"
    if value < -1.0:
        return "hyperboloid"
    if value < 0.0:
        return "prolate_ellipsoid"
    return "oblate_ellipsoid"


def camera_manifest_with_surface_adjustment(
    camera: CameraManifest | Mapping[str, Any] | None,
    surface_label: str,
    **adjustment: Any,
) -> CameraManifest:
    """Apply one sparse physical-surface edit to a canonical camera manifest."""

    label = str(surface_label).strip()
    if not label:
        raise ValueError("surface_label must be non-empty")
    if camera is None:
        current = resolve_camera_manifest().mapping()
    elif isinstance(camera, CameraManifest):
        current = camera.mapping()
    else:
        authored = dict(camera)
        current = resolve_camera_manifest(
            authored if "manifest" in authored else {"manifest": authored}
        ).mapping()
    current.pop("identity", None)
    current.pop("resolved", None)
    lens = dict(current.get("lens", {}))
    group_count = int(lens.get("group_count", 4))
    label_match = re.fullmatch(r"G([1-9][0-9]*)_(front|back)", label)
    if label_match is None or int(label_match.group(1)) > group_count:
        raise ValueError(
            f"surface_label must name a solved surface G1..G{group_count}_front/back"
        )
    adjustments = {
        str(key): dict(value)
        for key, value in dict(lens.get("surface_adjustments", {})).items()
    }
    allowed = {
        "axial_offset_mm", "shift_x_mm", "shift_y_mm",
        "tilt_x_deg", "tilt_y_deg", "radius_mm", "clear_radius_mm",
        "conic_type", "conic_k", "ior_after",
    }
    unknown = set(adjustment) - allowed
    if unknown:
        raise ValueError(f"unsupported surface adjustment fields {sorted(unknown)}")
    edited = dict(adjustments.get(label, {}))
    edited.update(adjustment)
    if "conic_k" in adjustment and "conic_type" not in adjustment:
        edited["conic_type"] = conic_type_from_k(adjustment["conic_k"])
    conic_type = edited.get("conic_type")
    if conic_type is not None:
        conic_type = str(conic_type).strip().lower()
        if conic_type not in CONIC_TYPE_DEFAULT_K:
            raise ValueError(f"unsupported conic_type {conic_type!r}")
        edited["conic_type"] = conic_type
        if "conic_k" not in adjustment:
            edited["conic_k"] = CONIC_TYPE_DEFAULT_K[conic_type]
    for key in allowed - {"conic_type"}:
        if key in edited:
            edited[key] = float(edited[key])
            if not math.isfinite(edited[key]):
                raise ValueError(f"surface {key} must be finite")
    if "clear_radius_mm" in edited and edited["clear_radius_mm"] <= 0.0:
        raise ValueError("surface clear_radius_mm must be positive")
    if "radius_mm" in edited and abs(edited["radius_mm"]) <= 1.0e-9:
        raise ValueError("surface radius_mm must be non-zero")
    if "ior_after" in edited and edited["ior_after"] < 1.0:
        raise ValueError("surface ior_after must be >= 1")
    adjustments[label] = edited
    lens["surface_adjustments"] = adjustments
    current["lens"] = lens
    return resolve_camera_manifest({"manifest": current})


def camera_manifest_with_controls(
    camera: CameraManifest | Mapping[str, Any] | None,
    *,
    zoom: float,
    focus_distance_m: float,
) -> CameraManifest:
    """Return a canonical manifest with editor zoom/focus controls applied."""

    if camera is None:
        current = resolve_camera_manifest().mapping()
    elif isinstance(camera, CameraManifest):
        current = camera.mapping()
    else:
        authored = dict(camera)
        current = resolve_camera_manifest(
            authored if "manifest" in authored else {"manifest": authored}
        ).mapping()
    current.pop("identity", None)
    current.pop("resolved", None)
    lens = dict(current.get("lens", {}))
    focus = dict(current.get("focus", {}))
    zoom = max(0.0, min(1.0, float(zoom)))
    focal_range = lens.get("focal_length_range_mm", [55.0, 110.0])
    if isinstance(focal_range, (list, tuple)) and len(focal_range) == 2:
        lens["focal_length_mm"] = (
            float(focal_range[0])
            + (float(focal_range[1]) - float(focal_range[0])) * zoom
        )
    lens["zoom"] = zoom
    focus["distance_m"] = max(1.0e-4, float(focus_distance_m))
    current["lens"] = lens
    current["focus"] = focus
    return resolve_camera_manifest({"manifest": current})


def camera_manifest_to_designer_preset(
    camera: CameraManifest | Mapping[str, Any] | None = None,
):
    """Return a designer preset derived from the production camera manifest.

    Optical-design coordinates run from object to sensor along ``+x``.  The
    existing designer faces the scene along ``-z``; consequently positions are
    mirrored about the fixed sensor plane while all physical distances remain
    unchanged.  This is a coordinate adapter, not another lens solve.
    """

    if camera is None:
        manifest = resolve_camera_manifest()
    elif isinstance(camera, CameraManifest):
        manifest = camera
    else:
        authored = dict(camera)
        manifest = resolve_camera_manifest(
            authored if "manifest" in authored else {"manifest": authored}
        )

    data = manifest.mapping()
    lens = dict(data["lens"])
    focus = dict(data["focus"])
    sensor = dict(data["sensor"])
    spectral = dict(data["spectral"])
    surface_adjustments = {
        str(key): dict(value)
        for key, value in dict(lens.get("surface_adjustments", {})).items()
    }
    focal_range_mm = lens.get("focal_length_range_mm", [55.0, 110.0])
    if not isinstance(focal_range_mm, (list, tuple)) or len(focal_range_mm) != 2:
        raise ValueError("camera lens focal_length_range_mm must contain two values")

    sensor_width_m = float(sensor["physical_width_mm"]) * 1.0e-3
    sensor_height_m = float(sensor["physical_height_mm"]) * 1.0e-3
    sensor_half_extent_m = 0.5 * max(sensor_width_m, sensor_height_m)
    spec = OpticalDesignSpec(
        focal_length_range_m=(
            float(focal_range_mm[0]) * 1.0e-3,
            float(focal_range_mm[1]) * 1.0e-3,
        ),
        zoom=float(lens.get("zoom", 0.5)),
        focus_distance_m=float(focus.get("distance_m", 1.0)),
        f_number=float(lens.get("f_number", 4.0)),
        entrance_x_m=1.08,
        sensor_x_m=1.25,
        sensor_clearance_m=0.030,
        image_radius_m=0.5 * float(sensor["image_circle_diameter_mm"]) * 1.0e-3,
        group_count=int(lens.get("group_count", 4)),
        default_ior=float(lens.get("default_ior", 1.55)),
        min_air_gap_m=float(lens.get("minimum_air_gap_mm", 8.0)) * 1.0e-3,
        group_thickness_m=float(lens.get("group_thickness_mm", 18.0)) * 1.0e-3,
        max_group_radius_m=float(lens.get("maximum_group_radius_mm", 120.0)) * 1.0e-3,
    )
    solved = solve_four_group_zoom_surrogate(spec)

    from camera_designer.camera_preset import (
        BodySpec,
        CameraPreset,
        GLASS_CATALOG,
        GlassSpec,
        LensElement,
        LensGroup,
    )
    from camera_designer.parametric_surfaces import (
        ApertureStop,
        ConicSurface,
        LensMountRing,
        SensorSurface,
    )

    glass_name = str(lens.get("glass", "N-BK7"))
    glass_catalog_key = {
        "N-BK7": "BK7",
        "NBK7": "BK7",
        "BK-7": "BK7",
    }.get(glass_name.upper(), glass_name)
    catalog_glass = GLASS_CATALOG.get(glass_catalog_key)
    elements: list[LensElement] = []
    for group in solved.groups:
        center_z = float(spec.sensor_x_m - group.x_m)
        half_thickness = 0.5 * float(group.thickness_m)
        # Preserve the authored glass law in the designer projection. A
        # constant group IOR makes the preview achromatic even though the
        # production MatBuf is wavelength-dependent, hiding chromatic focus
        # and making ray diagnostics disagree with native transport.
        glass = (
            GlassSpec(
                glass_name,
                n_d=float(catalog_glass.n_d),
                V_d=float(catalog_glass.V_d),
                B=tuple(catalog_glass.B),
                C=tuple(catalog_glass.C),
            )
            if catalog_glass is not None
            else GlassSpec(glass_name, n_d=float(group.ior))
        )
        air = GlassSpec("air", n_d=1.0)
        front_label = f"{group.name}_front"
        back_label = f"{group.name}_back"

        def _surface_element(label, *, radius, z_vertex, glass_out):
            edit = surface_adjustments.get(label, {})
            conic_type = str(edit.get("conic_type", "sphere")).strip().lower()
            conic_k = float(edit.get(
                "conic_k", CONIC_TYPE_DEFAULT_K.get(conic_type, 0.0)
            ))
            edited_glass = glass_out
            if "ior_after" in edit:
                edited_glass = GlassSpec(
                    getattr(glass_out, "name", "edited_glass"),
                    n_d=float(edit["ior_after"]),
                )
            return LensElement(
                surface=ConicSurface(
                    R=float(edit.get("radius_mm", radius * 1.0e3)) * 1.0e-3,
                    K=conic_k,
                    r_max=float(edit.get(
                        "clear_radius_mm", group.aperture_radius_m * 1.0e3
                    )) * 1.0e-3,
                ),
                glass_out=edited_glass,
                z_vertex=float(z_vertex) + float(edit.get("axial_offset_mm", 0.0)) * 1.0e-3,
                label=label,
                shift_xy_m=(
                    float(edit.get("shift_x_mm", 0.0)) * 1.0e-3,
                    float(edit.get("shift_y_mm", 0.0)) * 1.0e-3,
                ),
                tilt_xy_deg=(
                    float(edit.get("tilt_x_deg", 0.0)),
                    float(edit.get("tilt_y_deg", 0.0)),
                ),
            )

        elements.extend((
            _surface_element(
                front_label, radius=float(group.radius_front_m),
                z_vertex=center_z + half_thickness,
                glass_out=glass,
            ),
            _surface_element(
                back_label, radius=-float(group.radius_back_m),
                z_vertex=center_z - half_thickness,
                glass_out=air,
            ),
        ))

    wavelengths_um = [
        float(value) * 1.0e-3
        for value in spectral.get("active_wavelengths_nm", (460.0, 550.0, 640.0))
    ]
    aperture_z = float(spec.sensor_x_m - solved.aperture_x_m)
    aperture_radius = float(solved.aperture_radius_m)
    return CameraPreset(
        name=str(data.get("asset_key", "canonical-optical-camera")),
        body=BodySpec(
            width=max(0.10, sensor_width_m + 0.06),
            height=max(0.10, sensor_height_m + 0.05),
            depth=max(0.08, float(solved.assembly_back_x_m - solved.assembly_front_x_m) + 0.05),
            label="canonical 6x6 body",
        ),
        mount_ring=LensMountRing(
            z_flange=max(aperture_z + 0.010, 0.010),
            r_inner=max(aperture_radius * 1.15, sensor_half_extent_m),
            r_outer=max(aperture_radius * 1.45, sensor_half_extent_m + 0.008),
            back_clearance=0.004,
        ),
        lens_group=LensGroup(
            elements=elements,
            label=str(lens.get("profile", "canonical four-group lens")),
        ),
        aperture_stop=ApertureStop(
            z_pos=aperture_z,
            r_inner=0.0,
            r_outer=aperture_radius,
            n_blades=int(lens.get("aperture_blades", 8)),
        ),
        sensor=SensorSurface(
            z_pos=0.0,
            r_max=sensor_half_extent_m,
            pixel_pitch=(
                sensor_width_m
                / max(1, int(sensor.get("default_final_raster_px", [1024])[0]))
            ),
        ),
        focal_mm=float(lens.get("focal_length_mm", spec.target_focal_length_m * 1.0e3)),
        f_number=float(lens.get("f_number", 4.0)),
        wavelengths=wavelengths_um,
    )


__all__ = [
    "camera_manifest_to_designer_preset",
    "camera_manifest_with_controls",
    "camera_manifest_with_surface_adjustment",
    "CONIC_TYPE_DEFAULT_K",
    "conic_type_from_k",
]
