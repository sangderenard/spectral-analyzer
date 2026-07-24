from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wave_transform_visual_demo import (
    _arena_probe_panel,
    _component_static_panels,
    _aperture_sweep_radius,
    _aperture_spectral_samples,
    _remove_piston_phase,
    _signed_scalar_rgba,
    render_aperture_bake,
    render_sequence,
    run_aperture_live,
    run_component_arena_live,
    run_live,
    run_transport_live,
    ultra_bake_presets,
)


def test_visual_demo_uses_production_transform_and_writes_sequence(tmp_path):
    result = render_sequence(tmp_path, size=16, frames=2, scale=1)
    assert len(result["frames"]) == 3
    assert result["max_roundtrip_error"] < 2.0e-5
    for value in (*result["frames"], result["summary"]):
        path = Path(value)
        assert path.is_file()
        with Image.open(path) as image:
            assert image.width > 0
            assert image.height > 0


def test_live_mode_validates_before_opening_a_context():
    with pytest.raises(ValueError, match="power of two"):
        run_live(size=15)
    with pytest.raises(ValueError, match="cycle_steps"):
        run_live(cycle_steps=0)
    with pytest.raises(ValueError, match="fps"):
        run_live(fps=0)
    with pytest.raises(ValueError, match="panel_size"):
        run_transport_live(panel_size=32)
    with pytest.raises(ValueError, match="power of two"):
        run_aperture_live(size=15)
    with pytest.raises(ValueError, match="polarization"):
        run_aperture_live(size=16, polarization_mode="invented")
    with pytest.raises(ValueError, match="quality"):
        run_aperture_live(size=16, quality="reckless")
    with pytest.raises(ValueError, match="spectral_mode"):
        run_aperture_live(size=16, spectral_mode="quantized")
    with pytest.raises(ValueError, match="lane_count"):
        run_aperture_live(size=16, lane_count=2)
    with pytest.raises(ValueError, match="aperture_pattern"):
        run_aperture_live(size=16, aperture_pattern="ideal-mask")
    with pytest.raises(ValueError, match="size/fps"):
        run_component_arena_live("mirror.plane", size=32)
    with pytest.raises(ValueError, match="solve_hz"):
        run_component_arena_live("mirror.plane", solve_hz=0.0)


@pytest.mark.parametrize(
    "component_key",
    ("lens.default-camera", "mirror.plane", "pentaprism.finder"),
)
def test_component_arena_panels_use_compiled_component_artifacts(component_key):
    from camera_software.optical_components import (
        default_optical_component_registry,
    )

    registry = default_optical_component_registry()
    component = registry.create(component_key, 4)
    compiled = component.compile(4)
    panels, labels = _component_static_panels(
        component, compiled, 128, phase=0.37
    )

    assert len(panels) == len(labels) == 6
    assert all(panel.shape == (128, 128, 4) for panel in panels)
    assert np.count_nonzero(panels[1][..., :3]) > 0
    assert "GRAPH" in labels


def test_native_probe_panel_encodes_measured_spectrum_and_field_power():
    from camera_software.optical_components import PlaneMirrorComponent

    component = PlaneMirrorComponent()
    compiled = component.compile(4)
    dim = {
        "points": np.asarray(((-0.02, 0.0, 0.0), (0.0, 0.0, 0.0))),
        "rgb": np.asarray((0.1, 0.3, 1.0)),
        "relative_power": 0.04,
    }
    bright = {
        "points": np.asarray(((0.0, 0.0, 0.0), (-0.02, 0.01, 0.0))),
        "rgb": np.asarray((1.0, 0.2, 0.05)),
        "relative_power": 1.0,
    }

    panel = _arena_probe_panel(
        component, compiled, 128, 0.0, [dim, bright]
    )

    assert panel.shape == (128, 128, 4)
    assert np.max(panel[..., 0]) > 200
    assert np.max(panel[..., 2]) > 200


def test_relative_phase_gauge_removes_only_global_piston():
    y, x = np.mgrid[-1.0:1.0:9j, -1.0:1.0:9j]
    base = np.exp(-(x*x+y*y))*np.exp(1j*(0.7*x-0.35*y+0.2*x*y))
    rotated = base*np.exp(1j*2.17)
    base_relative, _ = _remove_piston_phase(base)
    rotated_relative, piston = _remove_piston_phase(rotated)

    np.testing.assert_allclose(
        rotated_relative, base_relative, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(np.abs(rotated_relative), np.abs(rotated))
    assert np.isfinite(piston)


def test_aperture_sweep_reaches_pinhole_and_clear_field_extremes():
    pitch = 0.75e-6
    minimum, assembly, sweep_min = _aperture_sweep_radius(0.0, 64, pitch)
    maximum, _, sweep_max = _aperture_sweep_radius(np.pi, 64, pitch)
    field_radius = 0.5*63*pitch

    assert minimum == pytest.approx(0.76*pitch)
    assert maximum == pytest.approx(1.51*field_radius)
    assert maximum < assembly
    assert sweep_min == pytest.approx(0.0)
    assert sweep_max == pytest.approx(1.0)


def test_aperture_fixed_and_continuous_spectra_share_exact_lane_widths():
    fixed_wavelengths, fixed_amplitudes = _aperture_spectral_samples(
        "fixed", 4, wavelength_m=532.0e-9,
    )
    continuous_a, continuous_amplitudes = _aperture_spectral_samples(
        "continuous", 4, wavelength_m=532.0e-9, sample_epoch=0,
    )
    continuous_b, _ = _aperture_spectral_samples(
        "continuous", 4, wavelength_m=532.0e-9, sample_epoch=1,
    )

    assert fixed_wavelengths.shape == (4,)
    assert continuous_a.shape == (4,)
    assert np.sum(fixed_amplitudes**2) == pytest.approx(1.0)
    assert np.sum(continuous_amplitudes**2) == pytest.approx(1.0)
    assert not np.allclose(
        continuous_a, continuous_b, rtol=1.0e-6, atol=1.0e-12
    )


def test_ultra_bake_preset_writes_plate_hero_and_manifest(tmp_path):
    result = render_aperture_bake(
        tmp_path,
        preset="iris-spectrum-continuous",
        size=8,
        frames=2,
        scale=1,
        quality="balanced",
        lane_count=3,
        aperture_pattern="circular",
        output_size=256,
    )

    assert len(result["frames"]) == 2
    assert Path(result["hero"]).is_file()
    manifest_path = Path(result["manifest"])
    assert manifest_path.is_file()
    manifest = __import__("json").loads(manifest_path.read_text())
    assert manifest["spectral_mode"] == "continuous"
    assert manifest["lane_count"] == 3
    assert manifest["aperture_pattern"] == "circular"
    assert manifest["output_size"] == 256
    assert manifest["publication_layout"] == (
        "aspect-preserving-square-texture"
    )
    assert len(manifest["frames"]) == 2
    with Image.open(result["frames"][0]) as image:
        assert image.size == (256, 256)
        assert image.mode == "RGBA"


def test_ultra_bake_recipes_are_detached_and_validate():
    recipes = ultra_bake_presets()
    assert set(recipes) == {
        "iris-spectrum-fixed",
        "iris-spectrum-continuous",
        "iris-polarization",
        "iris-coherent-phase",
    }
    recipes["iris-spectrum-fixed"]["size"] = 1
    assert ultra_bake_presets()["iris-spectrum-fixed"]["size"] == 64
    with pytest.raises(ValueError, match="unknown ultra bake"):
        render_aperture_bake(".", preset="made-up")


def test_ultra_bake_can_publish_one_unscaled_scientific_square(tmp_path):
    result = render_aperture_bake(
        tmp_path,
        preset="iris-spectrum-fixed",
        size=8,
        frames=1,
        scale=1,
        quality="balanced",
        lane_count=1,
        panel="spectral",
    )
    manifest = __import__("json").loads(
        Path(result["manifest"]).read_text()
    )

    with Image.open(result["hero"]) as image:
        assert image.size == (8, 8)
    assert manifest["publication_layout"] == "single-scientific-panel"
    assert manifest["frames"][0]["selected_panel_label"] == (
        "SPECTRAL POWER / RGB"
    )


def test_signed_stokes_display_opacity_follows_beam_support():
    y, x = np.mgrid[-1.0:1.0:17j, -1.0:1.0:17j]
    support = np.exp(-8.0*(x*x+y*y))
    rgba = _signed_scalar_rgba(support, support)

    assert rgba[8, 8, 0] > rgba[8, 8, 2]
    assert rgba[8, 8, 3] == 255
    assert rgba[0, 0, 3] < 8
