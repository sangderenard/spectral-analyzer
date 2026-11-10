"""Cold, source-preserving contract generation; no GL context or GPU work."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from camera_software.lens_contracts import (
    compound_contract, export_collection, profile_contract, profile_shader_parameters,
)
from camera_designer.compound_optics import ApertureStop, CompoundLens, ConicSurface

ROOT = Path(__file__).resolve().parents[1]


def test_discovers_all_profiles_and_preserves_every_authored_field():
    result = export_collection()
    paths = sorted((ROOT / "configs/lenses").glob("*.yaml"))
    assert len(result["contracts"]) == len(paths) == 7
    assert result["failures"] == [] and not result["installed_in_studio"]
    for path, contract in zip(paths, result["contracts"]):
        assert contract["source_profile"] == yaml.safe_load(path.read_bytes())
        assert contract["calibration"]["status"] == "authored-not-measured"
        assert contract["effects"]["transmission_wavelengths_nm"] is None
        assert contract["controls"]["focus_distance"]["maximum"] is None
        assert not contract["controls"]["focal_length"]["adjustable"]
    assert json.dumps(result, allow_nan=False) == json.dumps(export_collection(), allow_nan=False)


def test_macro_controls_retain_units_and_do_not_invent_two_centimetre_focus():
    contract = profile_contract(ROOT / "configs/lenses/macro_50mm.yaml")
    controls = contract["controls"]
    assert controls["focal_length"]["default"] == .05
    assert controls["focus_distance"]["minimum"] == .23
    assert controls["aperture"]["minimum"] == 2.8
    params = profile_shader_parameters(contract, focal_length_m=.05, focus_distance_m=.23,
                                      f_number=4., sensor_width_m=.036, sensor_height_m=.024)
    assert params["aperture_radius_m"] == pytest.approx(.00625)
    assert params["fov_y_degrees"] == pytest.approx(26.99146656)
    assert params["focus_reference_is_approximation"] is True
    assert params["installed_in_studio"] is False
    for change in ({"focus_distance_m": .019}, {"focal_length_m": .051}, {"f_number": 2.}):
        values = dict(focal_length_m=.05, focus_distance_m=.23, f_number=4.,
                      sensor_width_m=.036, sensor_height_m=.024)
        with pytest.raises(ValueError):
            profile_shader_parameters(contract, **{**values, **change})


@pytest.mark.parametrize("key,bad", [("focal_mm", True), ("min_focus_m", -1),
                                   ("max_aperture_fstop", float("nan")),
                                   ("transmission", [1, 1, 1]), ("is_zoom", "false")])
def test_malformed_profiles_fail_closed(tmp_path, key, bad):
    value = yaml.safe_load((ROOT / "configs/lenses/standard_35mm.yaml").read_bytes())
    value[key] = bad
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        profile_contract(path)
    result = export_collection(tmp_path)
    assert not result["contracts"] and len(result["failures"]) == 1


def test_hash_changes_when_source_changes_without_silently_reusing_old_contract(tmp_path):
    path = tmp_path / "lens.yaml"
    path.write_bytes((ROOT / "configs/lenses/standard_35mm.yaml").read_bytes())
    first = profile_contract(path)
    path.write_bytes(path.read_bytes() + b"\n# new source revision\n")
    second = profile_contract(path)
    assert first["source_sha256"] != second["source_sha256"]
    assert first["source_profile"] == second["source_profile"]


def biconvex():
    lens = CompoundLens()
    lens.add(ConicSurface(x_pos=0., R_curvature=.05, n_before=1., n_after=1.5, aperture_r=.01))
    lens.add(ConicSurface(x_pos=.005, R_curvature=-.05, n_before=1.5, n_after=1., aperture_r=.01))
    return lens


def test_compound_equivalent_uses_actual_surfaces_and_independent_exact_probes():
    lens = biconvex()
    contract = compound_contract(lens, source="test:biconvex", wavelength_nm=587.6)
    cardinal = contract["cardinal"]
    assert cardinal["image_effective_focal_length_m"] == pytest.approx(1. / (20. - 1./3.))
    assert cardinal["front_principal_plane_x_m"] > 0.
    assert cardinal["rear_principal_plane_x_m"] < .005
    assert contract["calibration"]["status"] == "paraxial-probe-passed"
    assert contract["eligible_for_air_to_air_positive_thin_lens"]
    assert contract["focus"]["validated_actuation_range_m"] is None
    near = contract["focus"]["conjugate_estimates"][1]
    assert near["object_distance_from_front_vertex_m"] == .01
    assert not near["real_image_after_last_surface"]
    far = contract["focus"]["conjugate_estimates"][-1]
    assert far["real_image_after_last_surface"]
    assert json.dumps(contract, allow_nan=False)


def test_scalar_reference_does_not_invent_wavelength_and_failed_probe_is_not_eligible():
    lens = biconvex()
    contract = compound_contract(lens, source="test:scalar")
    assert contract["reference_wavelength_nm"] is None
    # Corrupt the claimed paraxial matrix while retaining exact surfaces.
    lens._paraxial_matrix = lambda: np.eye(2)
    bad = compound_contract(lens, source="test:mismatch")
    assert bad["calibration"]["status"] == "paraxial-probe-failed"
    assert not bad["eligible_for_air_to_air_positive_thin_lens"]


def test_plane_separation_is_not_discarded_into_a_single_nominal_focal_number():
    lens = biconvex()
    first = compound_contract(lens, source="test:thin")
    lens.elements[-1].x_pos = .010
    second = compound_contract(lens, source="test:thicker")
    assert first["prescription_sha256"] != second["prescription_sha256"]
    assert first["cardinal"] != second["cardinal"]


def test_stop_outside_glass_is_not_mislabeled_as_last_refracting_vertex():
    lens = biconvex()
    before = compound_contract(lens, source="without-stop")
    lens.add(ApertureStop(x_pos=.015, r_clear=.01))
    after = compound_contract(lens, source="with-stop")
    assert after["matrix_reference_planes_x_m"] == [0., .015]
    assert after["cardinal"]["last_vertex_x_m"] == .005
    for name in ("back_focal_distance_m", "rear_principal_plane_x_m", "image_effective_focal_length_m"):
        assert after["cardinal"][name] == pytest.approx(before["cardinal"][name])


def test_camera_registry_is_automatically_preserved_without_claiming_named_equivalence():
    from camera_designer.camera_preset import PRESET_REGISTRY
    result = export_collection(include_camera_presets=True)
    entries = [r for r in result["contracts"] if "source_prescription" in r]
    assert len(entries) == len(PRESET_REGISTRY) == 2
    assert result["failures"] == []
    for entry in entries:
        key = entry["source"].rsplit(":", 1)[-1]
        assert entry["source_prescription"] == PRESET_REGISTRY[key].to_dict()
        assert entry["reference_wavelength_nm"] == pytest.approx(587.)
        assert entry["conversion"]["axial_scale"] == -1.
        assert not entry["accepted_as_named_camera_preset"]
        assert not entry["installed_in_studio"]
    assert json.dumps(result, allow_nan=False)


def test_duplicate_profile_names_fail_instead_of_replacing_another_lens(tmp_path):
    raw = (ROOT / "configs/lenses/standard_35mm.yaml").read_bytes()
    (tmp_path / "one.yaml").write_bytes(raw)
    (tmp_path / "two.yaml").write_bytes(raw)
    result = export_collection(tmp_path)
    assert len(result["contracts"]) == 1
    assert "duplicate" in result["failures"][0]["error"]


def test_existing_bench_actuator_is_probed_without_mutation_or_false_macro_claim():
    from camera_software.lens_contracts import _bench_focus_probes
    from thick_lens_focus_lab import SceneConfig, _compound_lens_from_scene
    scene = SceneConfig()
    lens = _compound_lens_from_scene(scene)
    before = lens.build_gpu_payload().copy()
    stack = list(scene.lens_stack)
    result = _bench_focus_probes(scene)
    assert scene.lens_stack == stack
    np.testing.assert_array_equal(_compound_lens_from_scene(scene).build_gpu_payload(), before)
    assert all(not r["available"] for r in result if r["focus_distance_from_front_vertex_m"] <= .02)
    focused = next(r for r in result if r["focus_distance_from_front_vertex_m"] == 1.)
    assert focused["available"] and focused["exact_probe_count"] == 4
    assert focused["exact_near_axis_max_sensor_radius_m"] <= 2e-6


def test_mirror_nominal_specs_do_not_invent_optics():
    contract = profile_contract(ROOT / "configs/lenses/mirror_500mm_f8.yaml")
    assert contract["controls"]["aperture"]["minimum"] == 8
    assert contract["controls"]["aperture"]["maximum"] == 8
    assert contract["controls"]["focus_distance"]["minimum"] is None
    assert contract["effects"]["distortion_k1"] is None
    assert contract["effects"]["transmission"] is None
    params = dict(focal_length_m=.5, focus_distance_m=5, f_number=8, sensor_width_m=.036, sensor_height_m=.024)
    assert profile_shader_parameters(contract, **params)["aperture_radius_m"] == .03125
    for aperture in (7.9, 8.1):
        with pytest.raises(ValueError):
            profile_shader_parameters(contract, **{**params, "f_number": aperture})


def test_legacy_lens_spec_refuses_invented_defaults_for_specs_only_profile():
    import ast
    from dataclasses import dataclass
    tree = ast.parse((ROOT / "demo_pluck_gl.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LensSpec")
    scope = {"dataclass": dataclass}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "LensSpec", "exec"), scope)
    with pytest.raises(ValueError, match="Specs-only"):
        scope["LensSpec"].from_dict({"specs_only": True, "name": "mirror_500mm_f8"})
