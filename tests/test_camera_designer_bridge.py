import pytest

from camera_software.camera_designer_bridge import (
    camera_manifest_to_designer_preset,
    camera_manifest_with_controls,
    camera_manifest_with_surface_adjustment,
)
from camera_software.camera_manifest import resolve_camera_manifest


def test_designer_uses_solved_canonical_six_by_six_camera():
    manifest = resolve_camera_manifest()
    preset = camera_manifest_to_designer_preset(manifest)

    assert preset.name == manifest.mapping()["asset_key"]
    assert len(preset.lens_group.elements) == 8
    assert preset.sensor.z_pos == pytest.approx(0.0)
    assert preset.sensor.r_max == pytest.approx(0.028)
    assert preset.focal_mm == pytest.approx(82.5)
    assert preset.f_number == pytest.approx(4.0)
    assert preset.aperture_stop.n_blades == 8
    assert len(preset.wavelengths) == 8
    glass = preset.lens_group.elements[0].glass_out
    assert glass.name == "N-BK7"
    assert glass.n_at(0.400) > glass.n_at(0.700)


def test_designer_bridge_tracks_manifest_focus_and_zoom():
    near = resolve_camera_manifest({
        "manifest": {
            "focus": {"distance_m": 0.5},
            "lens": {"zoom": 0.25},
        }
    })
    far = resolve_camera_manifest({
        "manifest": {
            "focus": {"distance_m": 4.0},
            "lens": {"zoom": 0.75},
        }
    })

    near_preset = camera_manifest_to_designer_preset(near)
    far_preset = camera_manifest_to_designer_preset(far)
    near_vertices = [item.z_vertex for item in near_preset.lens_group.elements]
    far_vertices = [item.z_vertex for item in far_preset.lens_group.elements]

    assert near_vertices != far_vertices


def test_camera_controls_resolve_back_into_manifest():
    resolved = camera_manifest_with_controls(
        resolve_camera_manifest(), zoom=0.75, focus_distance_m=2.5,
    )
    manifest = resolved.mapping()
    assert manifest["lens"]["zoom"] == pytest.approx(0.75)
    assert manifest["lens"]["focal_length_mm"] == pytest.approx(96.25)
    assert manifest["focus"]["distance_m"] == pytest.approx(2.5)
    assert camera_manifest_to_designer_preset(resolved).focal_mm == pytest.approx(96.25)


def test_surface_adjustment_is_canonical_and_changes_designer_geometry():
    base = resolve_camera_manifest()
    edited = camera_manifest_with_surface_adjustment(
        base,
        "G1_front",
        shift_x_mm=2.0,
        tilt_y_deg=4.0,
        radius_mm=140.0,
        clear_radius_mm=38.0,
        conic_type="hyperboloid",
        conic_k=-1.8,
    )
    assert edited.compatibility_key("optical_geometry") != base.compatibility_key(
        "optical_geometry"
    )
    surface = camera_manifest_to_designer_preset(edited).lens_group.elements[0]
    assert surface.label == "G1_front"
    assert surface.shift_xy_m == pytest.approx((0.002, 0.0))
    assert surface.tilt_xy_deg == pytest.approx((0.0, 4.0))
    assert surface.surface.R == pytest.approx(0.140)
    assert surface.surface.r_max == pytest.approx(0.038)
    assert surface.surface.K == pytest.approx(-1.8)


def test_conic_type_supplies_a_scientific_default_constant():
    edited = camera_manifest_with_surface_adjustment(
        resolve_camera_manifest(), "G2_back", conic_type="paraboloid"
    )
    adjustment = edited.mapping()["lens"]["surface_adjustments"]["G2_back"]
    assert adjustment["conic_k"] == pytest.approx(-1.0)


def test_direct_conic_constant_classifies_surface_and_bad_label_is_rejected():
    edited = camera_manifest_with_surface_adjustment(
        None, "G2_back", conic_k=-1.25,
    )
    adjustment = edited.mapping()["lens"]["surface_adjustments"]["G2_back"]
    assert adjustment["conic_type"] == "hyperboloid"
    with pytest.raises(ValueError, match="solved surface"):
        camera_manifest_with_surface_adjustment(None, "G9_front", tilt_x_deg=1.0)
