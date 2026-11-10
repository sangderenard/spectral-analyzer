"""Derive reviewable thin-camera contracts from existing lens assets.

This is a cold adapter, not a renderer or a replacement lens prescription.
Authored YAML profiles remain authored approximations. Exact CompoundLens
assets retain their prescription and gain a checked first-order equivalent.
No nominal label is treated as an optical measurement or focus certification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

SCHEMA = "derived-thin-lens-contract-v1"
ROOT = Path(__file__).resolve().parents[1]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile_contract(path: str | Path) -> dict[str, Any]:
    """Preserve the complete existing LensSpec YAML without filling gaps."""
    path = Path(path)
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("lens profile must be a YAML mapping")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("lens profile must contain finite JSON-compatible values") from exc
    specs_only = value.get("specs_only") is True
    required = ("name", "focal_mm", "min_focal_mm", "max_focal_mm", "is_zoom",
                "min_focus_m", "max_aperture_fstop", "distortion_k1",
                "distortion_k2", "vignetting", "transmission")
    optional = {"min_focus_m", "distortion_k1", "distortion_k2", "vignetting", "transmission"} if specs_only else set()
    missing = [key for key in required if key not in value and key not in optional]
    if missing:
        raise ValueError(f"lens profile is missing {missing}")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("lens name must be nonempty")
    if type(value["is_zoom"]) is not bool:
        raise ValueError("is_zoom must be boolean")
    numbers = {key: _number(value[key], key, positive=key in required[1:4]
                           or key in ("min_focus_m", "max_aperture_fstop"))
               for key in required if key not in ("name", "is_zoom", "transmission") and key in value}
    for key in optional - {"transmission"}:
        numbers.setdefault(key, None)
    lo, hi, focal = (numbers[key] for key in ("min_focal_mm", "max_focal_mm", "focal_mm"))
    if not lo <= focal <= hi or (not value["is_zoom"] and lo != hi):
        raise ValueError("focal range must contain the focal length; primes have one focal length")
    if numbers["vignetting"] is not None and not 0.0 <= numbers["vignetting"] <= 1.0:
        raise ValueError("vignetting must be in [0, 1]")
    transmission = value.get("transmission")
    if transmission is None and specs_only:
        pass
    elif not isinstance(transmission, list) or len(transmission) != 4:
        raise ValueError("existing LensSpec transmission requires four values")
    if any(not 0.0 <= _number(x, "transmission") <= 1.0 for x in (transmission or [])):
        raise ValueError("transmission values must be in [0, 1]")
    try:
        source = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source = path.name
    fixed_aperture = value.get("fixed_aperture", False)
    if type(fixed_aperture) is not bool:
        raise ValueError("fixed_aperture must be boolean")
    return {
        "schema": SCHEMA, "key": value["name"],
        "kind": "authored-thin-profile", "source": source,
        "source_sha256": _hash(raw), "source_profile": value,
        "calibration": {"status": "authored-not-measured", "surface_prescription": False},
        "controls": {
            "focal_length": {"unit": "m", "default": focal * 1e-3,
                             "minimum": lo * 1e-3, "maximum": hi * 1e-3,
                             "adjustable": bool(value["is_zoom"] and lo < hi)},
            "focus_distance": {"unit": "m", "minimum": numbers["min_focus_m"],
                               "maximum": None, "reference": "unspecified-in-source",
                               "bound_basis": "authored-profile-not-ray-validated"},
            "aperture": {"unit": "f-number", "minimum": numbers["max_aperture_fstop"],
                         "maximum": numbers["max_aperture_fstop"] if fixed_aperture else None, "pupil_relation": "diameter_m = focal_length_m / f_number"},
        },
        "effects": {
            "distortion_k1": numbers["distortion_k1"],
            "distortion_k2": numbers["distortion_k2"],
            "vignetting": numbers["vignetting"],
            "transmission": list(transmission) if transmission is not None else None,
            "transmission_wavelengths_nm": None,
            "coordinate_and_band_conventions": "not specified by the YAML; preserve until resolved",
        },
        "shader_adapter_ready": False,
        "limitations": [
            "This profile contains no glass surfaces or measured calibration.",
            "Do not infer spectral wavelengths, principal planes, working distance or a compound prescription.",
            "Distortion and vignetting coefficients are retained; their shader convention is not invented here.",
            "Unauthored focus/effect fields remain unknown; application travel limits are separate choices.",
            *(["Specs-only mirror lens: central obstruction, annular bokeh and mirror optics are not modeled."] if specs_only else []),
        ],
    }


def profile_shader_parameters(contract: Mapping[str, Any], *, focal_length_m: float,
                              focus_distance_m: float, f_number: float,
                              sensor_width_m: float, sensor_height_m: float) -> dict[str, Any]:
    """Resolve only supported analytical settings, without changing any renderer.

    The caller must explicitly adopt the approximate lens-plane focus convention.
    This cannot silently turn a source's unspecified focus datum into a physical
    sensor-to-object or front-glass working-distance guarantee.
    """
    if contract.get("schema") != SCHEMA or contract.get("kind") != "authored-thin-profile":
        raise ValueError("an authored thin profile contract is required")
    f = _number(focal_length_m, "focal_length_m", positive=True)
    u = _number(focus_distance_m, "focus_distance_m", positive=True)
    n = _number(f_number, "f_number", positive=True)
    w = _number(sensor_width_m, "sensor_width_m", positive=True)
    h = _number(sensor_height_m, "sensor_height_m", positive=True)
    controls = contract["controls"]
    zoom = controls["focal_length"]
    if not zoom["minimum"] <= f <= zoom["maximum"]:
        raise ValueError("focal length is outside the authored range")
    if controls["focus_distance"]["minimum"] is not None and u < controls["focus_distance"]["minimum"]:
        raise ValueError("focus distance is below the authored profile minimum")
    if n < controls["aperture"]["minimum"]:
        raise ValueError("aperture is wider than the authored maximum")
    if controls["aperture"].get("maximum") is not None and n > controls["aperture"]["maximum"]:
        raise ValueError("aperture is smaller than the authored fixed aperture")
    return {"contract_key": contract["key"], "source_sha256": contract["source_sha256"],
            "approximation": "authored-thin-profile", "focal_length_m": f,
            "focus_distance_m": u, "focus_reference": "analytical-lens-plane",
            "focus_reference_is_approximation": True, "f_number": n,
            "projection": "rectilinear analytical thin lens; not a calibrated fisheye mapping",
            "aperture_radius_m": f / (2.0 * n), "sensor_width_m": w,
            "sensor_height_m": h, "fov_y_degrees": math.degrees(2.0 * math.atan(h / (2.0 * f))),
            "fov_x_degrees": math.degrees(2.0 * math.atan(w / (2.0 * f))),
            "effects": dict(contract["effects"]),
            "effect_mapping_status": "retained, not applied; distortion coordinates and spectral bands unresolved",
            "installed_in_studio": False}


def compound_contract(lens: Any, *, source: str, wavelength_nm: float | None = None,
                      sensor_x_m: float | None = None) -> dict[str, Any]:
    """Derive cardinal data and independently probe the existing exact tracer.

    The contract is tied to one solved prescription/focus state and reference
    wavelength. It does not certify a focus motor's travel or a zoom interval.
    """
    from camera_designer.compound_optics import ApertureStop, TerminationReason

    if wavelength_nm is not None:
        wavelength_nm = _number(wavelength_nm, "wavelength_nm", positive=True)
    elements = list(lens.elements)
    if not elements:
        raise ValueError("compound lens has no optical elements")
    x0, x1 = float(elements[0].x_pos), float(elements[-1].x_pos)
    if any(float(b.x_pos) < float(a.x_pos) for a, b in zip(elements, elements[1:])):
        raise ValueError("compound prescription must be in canonical increasing-X order")
    refracting = [e for e in elements if not isinstance(e, ApertureStop)]
    if not refracting:
        raise ValueError("compound prescription has no refracting surfaces")
    front_vertex, last_vertex = float(refracting[0].x_pos), float(refracting[-1].x_pos)
    nin, nout = float(refracting[0].n_before), float(refracting[-1].n_after)
    m = np.asarray(lens._paraxial_matrix(), np.float64)
    if m.shape != (2, 2) or not np.isfinite(m).all():
        raise ValueError("invalid paraxial system matrix")
    a, b, c, d = m.ravel()
    powered = abs(c) > 1e-12
    h1 = x0 + nin * (d - 1.0) / c if powered else None
    h2 = x1 - nout * (a - 1.0) / c if powered else None
    fobj = -nin / c if powered else None
    fimg = -nout / c if powered else None
    # Trace from planes just outside the assembly. Same reduced-angle ABCD
    # convention as the existing paraxial solver; exact rays are independent.
    margin = max(1e-4, abs(x1 - x0) * .01)
    before, after = x0 - margin, x1 + margin
    t0 = np.array([[1., margin / nin], [0., 1.]])
    t1 = np.array([[1., margin / nout], [0., 1.]])
    predicted_matrix = t1 @ m @ t0
    probes = []
    for height in (-1e-5, 0., 1e-5):
        for angle in (-1e-4, 0., 1e-4):
            ray = lens.trace((before, height, 0.), (1., math.tan(angle), 0.))
            record = {"height_m": height, "angle_rad": angle,
                      "passed": ray.reason is TerminationReason.PASSED}
            if record["passed"] and abs(float(ray.direction[0])) > 1e-12:
                distance = (after - float(ray.origin[0])) / float(ray.direction[0])
                hit = ray.origin + distance * ray.direction
                actual = np.array([hit[1], nout * ray.direction[1] / ray.direction[0]])
                expected = predicted_matrix @ [height, nin * math.tan(angle)]
                record.update(height_error_m=float(abs(actual[0] - expected[0])),
                              reduced_angle_error=float(abs(actual[1] - expected[1])))
            else:
                record["passed"] = False
            probes.append(record)
    yerr = max((p.get("height_error_m", 1e30) for p in probes), default=1e30)
    aerr = max((p.get("reduced_angle_error", 1e30) for p in probes), default=1e30)
    paraxial_ok = all(p["passed"] for p in probes) and yerr <= 1e-8 and aerr <= 1e-6
    air = abs(nin - 1.) < 1e-6 and abs(nout - 1.) < 1e-6
    eligible = bool(paraxial_ok and air and powered and fimg > 0.)
    payload = np.ascontiguousarray(lens.build_gpu_payload(), np.float32)
    entrance_x, entrance_radius = (float(v) for v in lens.entrance_pupil)
    exit_x, exit_radius = (float(v) for v in lens.exit_pupil)
    pupil_values = (entrance_x, entrance_radius, exit_x, exit_radius)
    pupil_finite = all(math.isfinite(v) for v in pupil_values)
    conjugates = []
    # These are first-order conjugates, not measured focus limits. Preserve
    # their front-vertex reference so they cannot be mistaken for working
    # distance from a housing or for sensor-to-subject distance.
    for distance in (.005, .01, .02, .05, .1, .3, 1., 5.):
        object_x = front_vertex - distance
        denom = (-c - nin / (h1 - object_x)) if powered and abs(h1 - object_x) > 1e-12 else 0.
        image_x = h2 + nout / denom if powered and abs(denom) > 1e-12 else None
        conjugates.append({"object_distance_from_front_vertex_m": distance,
                           "image_x_m": float(image_x) if image_x is not None else None,
                           "real_image_after_last_surface": bool(image_x is not None and image_x > last_vertex)})
    return {
        "schema": SCHEMA, "key": source, "kind": "compound-paraxial-equivalent",
        "source": source, "prescription_sha256": _hash(payload.tobytes()),
        "reference_wavelength_nm": wavelength_nm,
        "index_basis": "resolved scalar prescription indices; wavelength unspecified" if wavelength_nm is None
                       else "caller-resolved scalar prescription at the declared wavelength",
        "axis": "canonical +X, object to sensor", "length_unit": "m",
        "surface_count": len(refracting), "stop_count": len(elements) - len(refracting),
        "matrix_convention": "[height_m, refractive_index * slope]",
        "matrix": m.tolist(), "matrix_reference_planes_x_m": [x0, x1],
        "object_medium_ior": nin, "image_medium_ior": nout,
        "cardinal": {"first_vertex_x_m": front_vertex, "last_vertex_x_m": last_vertex,
                     "front_principal_plane_x_m": h1, "rear_principal_plane_x_m": h2,
                     "object_effective_focal_length_m": fobj,
                     "image_effective_focal_length_m": fimg,
                     "back_focal_distance_m": x1 - last_vertex - nout * a / c if powered else None},
        "sensor_x_m": sensor_x_m,
        "pupils": {
            "basis": "existing CompoundLens paraxial pupil estimates; not full-field ray calibration",
            "entrance_x_m": entrance_x if pupil_finite else None,
            "entrance_radius_m": entrance_radius if pupil_finite else None,
            "exit_x_m": exit_x if pupil_finite else None,
            "exit_radius_m": exit_radius if pupil_finite else None,
            "f_number_estimate": fobj / (2.0 * entrance_radius)
                if powered and pupil_finite and entrance_radius > 0. else None,
        },
        "calibration": {"status": "paraxial-probe-passed" if paraxial_ok else "paraxial-probe-failed",
                        "height_tolerance_m": 1e-8, "reduced_angle_tolerance": 1e-6,
                        "max_height_error_m": yerr, "max_reduced_angle_error": aerr,
                        "probes": probes},
        "eligible_for_air_to_air_positive_thin_lens": eligible,
        "focus": {"validated_actuation_range_m": None, "conjugate_estimates": conjugates,
                  "datum": "front vertex; not front housing or sensor", "mechanical_travel_known": False},
        "limitations": [
            "One solved prescription and one reference index, not a measured zoom or focus sweep.",
            "Near-axis checks do not certify full-field distortion, vignetting, aperture throughput or bokeh.",
            "A positive conjugate does not prove the sensor or focus group can travel there.",
            "The current Studio shader does not consume principal-plane-separated contracts yet.",
            "Source prescription, exact transport, and wavelength sampling are unchanged.",
        ], "installed_in_studio": False, "shader_adapter_ready": False,
    }


def _bench_focus_probes(scene: Any) -> list[dict[str, Any]]:
    """Check the existing rear-group actuator bounds without moving a live scene.

    Unlike the live helper's end-stop fallback, an unbracketed focus request is
    reported as unavailable. These are sampled actuator checks, not an inferred
    continuous range or a full-aperture image-quality certification.
    """
    from thick_lens_focus_lab import _compound_lens_from_stack, _paraxial_image_x
    from camera_designer.compound_optics import TerminationReason

    stack = list(scene.lens_stack)
    iris = scene.iris_aperture
    front = float(stack[0].x_front)
    sensor = float(scene.image_plate.x)
    half = float(stack[-1].thickness) * .5
    lo = float(stack[-2].x_back) + half + float(getattr(scene, "_min_air_gap_m", .005))
    hi = sensor - half - .002
    rows = []
    for distance in (.005, .01, .02, .05, .1, .3, 1., 5., 10.):
        row = {"focus_distance_from_front_vertex_m": distance,
               "available": False, "rear_group_bounds_m": [lo, hi]}
        rows.append(row)
        if lo >= hi:
            row["reason"] = "no legal rear-group travel"
            continue
        object_x = front - distance

        def optics_at(position):
            updated = list(stack)
            updated[-1] = replace(stack[-1], center_x=position)
            return _compound_lens_from_stack(updated, iris)

        def error(position):
            return float(_paraxial_image_x(optics_at(position), object_x)) - sensor

        a, b = lo, hi
        ea, eb = error(a), error(b)
        if not math.isfinite(ea) or not math.isfinite(eb) or ea * eb > 0.:
            row["reason"] = "requested focus is not bracketed by the existing actuator"
            row["end_stop_residual_m"] = [ea if math.isfinite(ea) else None,
                                         eb if math.isfinite(eb) else None]
            continue
        for _ in range(48):
            mid = .5 * (a + b)
            em = error(mid)
            if not math.isfinite(em):
                break
            if abs(em) < 1e-9:
                a = b = mid
                break
            if ea * em <= 0.:
                b = mid
            else:
                a, ea = mid, em
        position = .5 * (a + b)
        residual = error(position)
        optics = optics_at(position)
        # Independent actual conic rays aimed near the axis. A pole in the
        # paraxial conjugate equation must never count as a focus solution.
        radius = float(stack[0].aperture_radius) * .001
        errors = []
        for y, z in ((radius, 0.), (-radius, 0.), (0., radius), (0., -radius)):
            ray = optics.trace((object_x, 0., 0.), (distance, y, z))
            if ray.reason is not TerminationReason.PASSED or ray.direction[0] <= 0.:
                continue
            travel = (sensor - float(ray.origin[0])) / float(ray.direction[0])
            if travel <= 0.:
                continue
            hit = ray.origin + travel * ray.direction
            errors.append(float(np.linalg.norm(hit[1:])))
        ok = math.isfinite(residual) and abs(residual) <= 5e-6 and len(errors) == 4 and max(errors) <= 2e-6
        row.update(available=bool(ok), rear_group_center_m=position,
                   sensor_plane_residual_m=residual if math.isfinite(residual) else None,
                   exact_near_axis_max_sensor_radius_m=max(errors) if errors else None,
                   exact_probe_count=len(errors),
                   reason="near-axis focus checks passed" if ok else "focus residual or exact ray checks failed")
    return rows


def camera_preset_contract(preset: Any, *, source: str, wavelength_um: float = .587) -> dict[str, Any]:
    """Export the existing SceneBuilder conversion without repairing its optics.

    CameraPreset front-to-sensor transport uses decreasing local Z, as in
    SceneBuilder's existing exact path. Keep the full serialized source and
    distinguish an unsupported image medium/curved sensor from an air camera.
    """
    from camera_designer.compound_optics import CompoundLens
    wavelength_um = _number(wavelength_um, "wavelength_um", positive=True)
    source_data = preset.to_dict()
    lens = CompoundLens.from_preset(preset, wavelength_um=wavelength_um, axial_scale=-1.)
    item = compound_contract(lens, source=source, wavelength_nm=wavelength_um * 1000.,
                             sensor_x_m=-float(preset.sensor.z_pos))
    item["source_prescription"] = source_data
    item["source_sha256"] = _hash(json.dumps(source_data, sort_keys=True, allow_nan=False).encode())
    item["conversion"] = {"entry": "CompoundLens.from_preset", "axial_scale": -1.,
                          "basis": "existing camera_designer.scene_builder exact transport"}
    item["authored_metadata"] = {"focal_length_m": float(preset.focal_mm) * .001,
                                 "f_number": float(preset.f_number), "fov_degrees": float(preset.fov_deg)}
    item["label_is_not_measurement"] = True
    item["accepted_as_named_camera_preset"] = False
    item["preset_fit"] = {"status": "not-approved",
                           "reason": "registry metadata and sensor/focus domain are not calibrated by this export"}
    item["limitations"].append("Registry surface conventions are preserved; negative power or a non-air image medium is not relabeled as a positive air thin lens.")
    return item


def export_collection(profile_dir: str | Path = ROOT / "configs" / "lenses",
                      *, include_bench_presets: bool = False,
                      include_camera_presets: bool = False) -> dict[str, Any]:
    """Discover profiles automatically; optionally solve every existing bench preset."""
    profiles, failures = [], []
    names = set()
    for path in sorted(Path(profile_dir).glob("*.yaml")):
        try:
            item = profile_contract(path)
            if item["key"] in names:
                raise ValueError(f"duplicate lens contract key {item['key']!r}")
            names.add(item["key"])
            profiles.append(item)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            failures.append({"source": path.name, "error": str(exc)})
    if include_camera_presets:
        from camera_designer.camera_preset import PRESET_REGISTRY
        for name, preset in sorted(PRESET_REGISTRY.items()):
            try:
                profiles.append(camera_preset_contract(preset, source=f"camera_designer.camera_preset.PRESET_REGISTRY:{name}"))
            except (ValueError, RuntimeError, AttributeError) as exc:
                failures.append({"source": f"camera-preset:{name}", "error": str(exc)})
    if include_bench_presets:
        from thick_lens_focus_lab import SceneConfig, _LENS_PRESETS_6x6, _compound_lens_from_scene, _paraxial_image_x
        for label, focal_range, fnumber, zoom in _LENS_PRESETS_6x6:
            try:
                scene = SceneConfig()
                scene.optical_design = replace(scene.optical_design,
                    focal_length_range_m=focal_range, f_number=fnumber, zoom=zoom)
                lens = _compound_lens_from_scene(scene)
                item = compound_contract(lens, source=f"thick_lens_focus_lab._LENS_PRESETS_6x6:{label}",
                                         sensor_x_m=float(scene.image_plate.x))
                item["authored_solver_input"] = asdict(scene.optical_design.spec)
                item["label_is_not_measurement"] = True
                spec = scene.optical_design.spec
                target = float(spec.target_focal_length_m)
                actual = item["cardinal"]["image_effective_focal_length_m"]
                focus_error = float(_paraxial_image_x(lens, float(scene.object_plane.x))) - float(scene.image_plate.x)
                focal_ok = actual is not None and abs(actual - target) <= float(spec.focal_length_tolerance_m)
                fit_ok = bool(focal_ok and math.isfinite(focus_error) and abs(focus_error) <= 5e-6)
                item["preset_fit"] = {
                    "status": "passed" if fit_ok else "failed",
                    "requested_focal_length_m": target, "actual_focal_length_m": actual,
                    "focal_tolerance_m": float(spec.focal_length_tolerance_m),
                    "actual_sensor_focus_residual_m": focus_error if math.isfinite(focus_error) else None,
                    "sensor_focus_tolerance_m": 5e-6,
                    "exact_refinement_accepted": getattr(scene, "_exact_fixed_back_design_id", None) == id(scene.optical_design),
                }
                item["controls"] = {
                    "focal_length": {"unit": "m", "requested_minimum": focal_range[0],
                                     "requested_maximum": focal_range[1], "sampled_zoom": zoom,
                                     "validated_zoom_interval": None},
                    "aperture": {"unit": "f-number", "authored_maximum_opening": fnumber,
                                 "validated_adjustment_range": None},
                    "focus_distance": {"unit": "m", "reference": "first refracting vertex",
                                       "mode": "solve and validate within rear-group travel",
                                       "validated_continuous_interval": None},
                }
                item["focus"]["mechanical_travel_known"] = True
                item["focus"]["sampled_actuator_checks"] = _bench_focus_probes(scene)
                item["focus"]["sampled_checks_are_not_continuous_domain"] = True
                item["accepted_as_named_camera_preset"] = bool(fit_ok and item["eligible_for_air_to_air_positive_thin_lens"])
                profiles.append(item)
            except (ValueError, RuntimeError, AttributeError) as exc:
                failures.append({"source": f"bench:{label}", "error": str(exc)})
    return {"schema": "lens-contract-collection-v1", "contracts": profiles,
            "failures": failures, "installed_in_studio": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "configs" / "lenses")
    parser.add_argument("--include-bench-presets", action="store_true")
    parser.add_argument("--include-camera-presets", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = export_collection(args.profile_dir, include_bench_presets=args.include_bench_presets,
                                   include_camera_presets=args.include_camera_presets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(collection['contracts'])} contracts; {len(collection['failures'])} source failures")
    return int(bool(collection["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
