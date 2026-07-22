"""Scrollable camera-loadout hierarchy for the selected rendered work."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json(path: str) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _resolved_order(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = _mapping(payload.get("defaults"))
    jobs = payload.get("jobs", ())
    job = _mapping(jobs[0]) if isinstance(jobs, (list, tuple)) and jobs else {}
    return defaults, _deep_merge(defaults, job)


def _latest_scene(payload: Mapping[str, Any]) -> Any | None:
    if str(payload.get("kind", "")) == "asset":
        subtype = payload.get("subtype")
        scenes = tuple(getattr(subtype, "scenes", ()))
        return scenes[-1] if scenes else None
    if str(payload.get("kind", "")) == "interface":
        assembly = payload.get("assembly")
        return getattr(assembly, "latest", None)
    return None


def _summary_candidate(scene: Any, scene_path: str) -> str:
    direct = str(getattr(scene, "resolved_summary_path", ""))
    if direct and os.path.isfile(direct):
        return direct
    root = Path(scene_path).parent if scene_path else None
    if root is not None and root.is_dir():
        direct_candidate = root / "0000_cpp_summary.json"
        if direct_candidate.is_file():
            return str(direct_candidate)
        for child in sorted(root.iterdir()):
            candidate = child / "0000_cpp_summary.json"
            if child.is_dir() and candidate.is_file():
                return str(candidate)
    return ""


def selected_camera_context(
    selected_payload: Mapping[str, Any] | None,
    fallback_order: Mapping[str, Any] | None = None,
    camera_profile: Any = None,
) -> dict[str, Any]:
    """Resolve authored and solved camera state for the current work selection."""

    payload = _mapping(selected_payload)
    scene = _latest_scene(payload)
    scene_path = str(getattr(scene, "scene_path", ""))
    order = _read_json(scene_path)
    if not order:
        order = _mapping(fallback_order)
    defaults, resolved = _resolved_order(order)
    camera = _deep_merge(
        _mapping(defaults.get("camera")),
        _mapping(resolved.get("camera")),
    )
    exposure = _deep_merge(
        _mapping(defaults.get("exposure")),
        _mapping(resolved.get("exposure")),
    )
    image = _deep_merge(
        _mapping(defaults.get("image")),
        _mapping(resolved.get("image")),
    )
    flash = _deep_merge(
        _mapping(defaults.get("flash")),
        _mapping(resolved.get("flash")),
    )
    if scene is not None:
        camera = _deep_merge(camera, _mapping(getattr(scene, "camera_spec", {})))
        exposure = _deep_merge(
            exposure, _mapping(getattr(scene, "exposure_spec", {}))
        )
    summary_path = _summary_candidate(scene, scene_path)
    summary = _read_json(summary_path)
    frame = _mapping(summary.get("frame_config_summary"))
    from camera_software.camera_manifest import resolve_camera_manifest
    authored_manifest = resolve_camera_manifest(
        camera, image, flash
    ).mapping()
    manifest = _deep_merge(
        authored_manifest, _mapping(frame.get("camera_manifest"))
    )
    profile = {}
    if camera_profile is not None:
        profile = {
            "sensor_white_level": getattr(camera_profile, "sensor_white_level", None),
            "exposure_compensation_ev": getattr(
                camera_profile, "exposure_compensation_ev", None
            ),
            "white_balance": getattr(camera_profile, "white_balance", None),
            "tone_curve_mode": getattr(camera_profile, "tone_curve_mode", ""),
            "output_space": getattr(camera_profile, "output_space", ""),
        }
    return {
        "selection_kind": str(payload.get("kind", "live") or "live"),
        "selection_label": str(
            getattr(payload.get("subtype"), "text", "")
            or getattr(payload.get("subtype"), "display_name", "")
            or payload.get("token", "")
            or getattr(scene, "display_name", "")
            or resolved.get("token", "")
            or "live work"
        ),
        "scene_path": scene_path,
        "summary_path": summary_path,
        "camera": camera,
        "exposure": exposure,
        "image": image,
        "flash": flash,
        "manifest": manifest,
        "frame": frame,
        "profile": profile,
        "has_solved_state": bool(frame) or bool(_mapping(manifest.get("resolved"))),
        "pose_animations": tuple(
            getattr(payload.get("subtype"), "pose_animations", ())
        ),
    }


def _number(value: Any, digits: int = 6, missing: str = "not authored") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return f"{number:.{digits}f}"


def _vector(value: Any, digits: int = 6) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "not authored"
    return "[" + ", ".join(_number(item, digits) for item in value) + "]"


def _first(source: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in source:
            return source[name]
    return default


def _movement_pair(
    camera: Mapping[str, Any],
    compound_names: tuple[str, ...],
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
    *,
    radians_to_degrees: bool = False,
) -> tuple[float, float]:
    compound = _first(camera, *compound_names)
    if isinstance(compound, (list, tuple)) and len(compound) >= 2:
        x, y = float(compound[0]), float(compound[1])
    else:
        x = float(_first(camera, *x_names, default=0.0) or 0.0)
        y = float(_first(camera, *y_names, default=0.0) or 0.0)
    if radians_to_degrees:
        x, y = math.degrees(x), math.degrees(y)
    return x, y


def camera_loadout_sections(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build stable expandable hierarchy rows from one resolved context."""

    camera = _mapping(context.get("camera"))
    exposure = _mapping(context.get("exposure"))
    image = _mapping(context.get("image"))
    frame = _mapping(context.get("frame"))
    profile = _mapping(context.get("profile"))
    manifest = _mapping(context.get("manifest"))
    manifest_identity = _mapping(manifest.get("identity"))
    compatibility = _mapping(manifest_identity.get("compatibility"))
    lens_manifest = _mapping(manifest.get("lens"))
    sensor_manifest = _mapping(manifest.get("sensor"))
    spectral_manifest = _mapping(manifest.get("spectral"))
    flash_manifest = _mapping(manifest.get("flash"))
    body_manifest = _mapping(manifest.get("body"))
    resolved_optics = _mapping(manifest.get("resolved"))
    focal = float(lens_manifest.get(
        "focal_length_mm", camera.get("focal_mm", 82.5)
    ) or 82.5)
    aperture = float(lens_manifest.get(
        "aperture_diameter_mm", camera.get("aperture_mm", 20.625)
    ) or 20.625)
    f_number = float(lens_manifest.get(
        "f_number", focal / aperture if aperture > 0.0 else float("nan")
    ))

    full_w = int(image.get("width", 0) or 0)
    full_h = int(image.get("height", 0) or 0)
    region = _mapping(image.get("region"))
    crop_x = int(region.get("x", 0) or 0)
    crop_y = int(region.get("y", 0) or 0)
    crop_w = int(region.get("width", full_w) or full_w)
    crop_h = int(region.get("height", full_h) or full_h)
    sensor_w_mm = float(sensor_manifest.get(
        "physical_width_mm", camera.get("sensor_w_mm", 56.0)
    ) or 56.0)
    sensor_h_mm = float(sensor_manifest.get(
        "physical_height_mm", camera.get("sensor_h_mm", 56.0)
    ) or 56.0)
    active_w_mm = sensor_w_mm * crop_w / full_w if full_w else sensor_w_mm
    active_h_mm = sensor_h_mm * crop_h / full_h if full_h else sensor_h_mm

    position = camera.get("position_m")
    target = camera.get("target_m")
    focus_target = camera.get("focus_target_m", target)
    focus_distance = camera.get("focus_distance_m")
    if focus_distance is None and isinstance(position, (list, tuple)) and isinstance(
        focus_target, (list, tuple)
    ) and len(position) == len(focus_target) == 3:
        focus_distance = math.sqrt(sum(
            (float(focus_target[i]) - float(position[i])) ** 2 for i in range(3)
        ))

    shift_x, shift_y = _movement_pair(
        camera,
        ("tilt_shift", "lens_shift_mm", "front_shift_mm"),
        ("lens_shift_x_mm", "front_shift_x_mm"),
        ("lens_shift_y_mm", "front_shift_y_mm"),
    )
    tilt_x, tilt_y = _movement_pair(
        camera,
        ("lens_tilt_deg", "front_tilt_deg"),
        ("lens_tilt_x_deg", "front_tilt_x_deg"),
        ("lens_tilt_y_deg", "front_tilt_y_deg"),
    )
    if "lens_tilt" in camera and "lens_tilt_deg" not in camera:
        tilt_x, tilt_y = _movement_pair(
            camera, ("lens_tilt",), (), (), radians_to_degrees=True
        )
    sensor_shift_x, sensor_shift_y = _movement_pair(
        camera,
        ("sensor_shift_mm",),
        ("sensor_shift_x_mm",),
        ("sensor_shift_y_mm",),
    )

    back = _mapping(camera.get("back"))
    back_type = str(
        camera.get("back_type")
        or back.get("type")
        or frame.get("camera_back_type")
        or sensor_manifest.get("back_type")
        or "flat"
    )
    solved = bool(context.get("has_solved_state"))
    source_note = "solved renderer state" if solved else "authored/default state"
    pose_animations = tuple(context.get("pose_animations", ()))
    preparation_cache_key = "not resolved"
    optical_payload_cache_key = "not resolved"
    if manifest_identity.get("manifest_hash"):
        from camera_software.camera_manifest import CameraManifest
        from camera_software.camera_preparation import CameraPreparationKey
        preparation = CameraPreparationKey.from_manifest(CameraManifest(manifest))
        preparation_cache_key = preparation.cache_key
        optical_payload_cache_key = preparation.optical_payload_cache_key

    sections = [
        {
            "key": "selection",
            "title": f"SELECTED  {context.get('selection_label', 'work')}",
            "lines": [
                f"kind: {context.get('selection_kind', 'live')}",
                f"source: {source_note}",
                f"asset key: {manifest.get('asset_key', 'camera/unknown')}",
                f"manifest: {manifest_identity.get('manifest_hash', 'not resolved')}",
                f"optics compatibility: {compatibility.get('optical_geometry', 'not resolved')}",
                f"prepared geometry cache: {preparation_cache_key}",
                f"optical payload cache: {optical_payload_cache_key}",
                f"scene: {context.get('scene_path') or 'live pending scene'}",
                f"solve summary: {context.get('summary_path') or 'not available yet'}",
            ],
            "accent": (198, 154, 65),
        },
        {
            "key": "camera",
            "title": "CAMERA BODY / POSE",
            "lines": [
                f"model: {camera.get('model', camera.get('camera_type', 'physical camera rig'))}",
                f"position m: {_vector(position)}",
                f"target m: {_vector(target)}",
                f"up: {_vector(camera.get('up'))}",
                f"camera mode: {frame.get('camera_mode_name', camera.get('mode', 'renderer default'))}",
                f"outer material: {body_manifest.get('outer_material', 'not authored')}",
                f"outer reflectance: {_number(body_manifest.get('outer_reflectance'), 6)}",
                f"outer roughness: {_number(body_manifest.get('outer_roughness'), 6)}",
                f"outer metallic: {_number(body_manifest.get('outer_metallic'), 6)}",
                f"reflection visibility: {body_manifest.get('reflection_visibility', 'not authored')}",
                f"interior material: {body_manifest.get('interior_material', 'not authored')}",
            ],
            "accent": (90, 140, 205),
        },
        {
            "key": "lens",
            "title": "LENS / APERTURE",
            "lines": [
                f"family: {lens_manifest.get('family', 'not authored')}",
                f"profile: {lens_manifest.get('profile', 'not authored')}",
                f"focal length: {_number(focal, 6)} mm",
                f"effective focal: {_number(resolved_optics.get('effective_focal_length_mm', frame.get('effective_focal_mm')), 6)} mm",
                f"aperture diameter: {_number(aperture, 6)} mm",
                f"f-number: f/{_number(f_number, 6)}",
                f"groups/surfaces: {lens_manifest.get('group_count', 'unknown')} / {2 * int(lens_manifest.get('group_count', 0) or 0)}",
                f"glass: {lens_manifest.get('glass', 'not authored')}",
                f"surface model: {lens_manifest.get('surface_model', 'not authored')}",
                f"representation: {lens_manifest.get('representation', 'not authored')}",
                f"design fidelity: {lens_manifest.get('design_fidelity', 'not authored')}",
                f"aperture radius solved: {_number(frame.get('camera_aperture_radius_mm'), 6)} mm",
                f"cone half-angle: {_number(frame.get('camera_cone_half_angle_deg'), 9)} deg",
                f"cone solid angle: {_number(frame.get('camera_cone_solid_angle_sr'), 12)} sr",
                f"aperture samples: {frame.get('camera_cone_aperture_samples', 'not solved')}",
                f"entrance pupil [x,r] m: {_vector(resolved_optics.get('entrance_pupil'), 9)}",
                f"exit pupil [x,r] m: {_vector(resolved_optics.get('exit_pupil'), 9)}",
                f"exit pupil kind: {resolved_optics.get('exit_pupil_kind', 'not solved')}",
                f"optical model: {'thin-lens planes' if frame.get('thin_lens_planes') else 'exact/parametric assembly'}",
            ],
            "accent": (110, 175, 220),
        },
        {
            "key": "focus",
            "title": "FOCUS / SOLVE",
            "lines": [
                f"focus target m: {_vector(focus_target)}",
                f"focus distance: {_number(focus_distance, 9)} m",
                f"focus mechanism: {_mapping(manifest.get('focus')).get('mechanism', 'not authored')}",
                f"solve status: {frame.get('camera_solve_status', 'not solved yet')}",
                f"solve error: {_number(frame.get('camera_solve_error'), 12)}",
                f"iterations: {frame.get('camera_solve_iter', 'not solved')}",
                f"circle of confusion: {_number(frame.get('camera_coc_um'), 6)} um",
                f"sensor adjustment: {_number(frame.get('camera_sensor_adjust_mm'), 9)} mm",
                f"thin-lens sensor z: {_number(frame.get('camera_thin_lens_target_sensor_z_m'), 12)} m",
                f"pinhole sensor z: {_number(frame.get('camera_pinhole_target_sensor_z_m'), 12)} m",
                f"pinhole comparison: {_number(frame.get('camera_pinhole_ref_mm'), 9)} mm",
            ],
            "accent": (75, 185, 145),
        },
        {
            "key": "movements",
            "title": "FINE MOVEMENTS / RAILS",
            "lines": [
                f"lens shift X: {shift_x:+.9f} mm",
                f"lens shift Y: {shift_y:+.9f} mm",
                f"lens tilt X: {tilt_x:+.9f} deg",
                f"lens tilt Y: {tilt_y:+.9f} deg",
                f"sensor shift X: {sensor_shift_x:+.9f} mm",
                f"sensor shift Y: {sensor_shift_y:+.9f} mm",
                f"front shift solved: {_number(frame.get('camera_front_shift_mm'), 9)} mm",
                f"front tilt solved: {_number(frame.get('camera_front_tilt_deg'), 9)} deg",
                f"sensor shift solved: {_number(frame.get('camera_sensor_shift_mm'), 9)} mm",
                f"film depth delta: {_number(camera.get('film_depth_delta_mm'), 9)} mm",
                f"film tilt right: {_number(camera.get('film_tilt_right_deg'), 9)} deg",
                f"film tilt up: {_number(camera.get('film_tilt_up_deg'), 9)} deg",
                f"rail ranges: {frame.get('camera_rail_ranges', 'renderer defaults')}",
                f"rail front shift: {_number(frame.get('camera_rail_usage_front_shift_pct'), 6)}%",
                f"rail front tilt: {_number(frame.get('camera_rail_usage_front_tilt_pct'), 6)}%",
                f"rail sensor shift: {_number(frame.get('camera_rail_usage_sensor_shift_pct'), 6)}%",
                f"rail corner use: {_number(frame.get('camera_rail_usage_corner_pct'), 6)}%",
                f"rail bellows use: {_number(frame.get('camera_rail_usage_bellows_pct'), 6)}%",
            ],
            "accent": (205, 115, 170),
        },
        {
            "key": "back",
            "title": "CAMERA BACK / FILM",
            "lines": [
                f"back type: {back_type}",
                f"back geometry: {back.get('geometry', camera.get('back_geometry', 'flat'))}",
                f"sensor/film slot: {camera.get('sensor_film_slot', 0)}",
                f"film/sensor name: {camera.get('film_name', camera.get('sensor_name', 'native spectral sensor'))}",
                f"optical provenance: {frame.get('camera_optical_provenance') or resolved_optics.get('prepared_build_id') or 'scene-order camera'}",
            ],
            "accent": (145, 115, 205),
        },
        {
            "key": "sensor",
            "title": "SENSOR / CROP",
            "lines": [
                f"physical sensor: {sensor_w_mm:.6f} x {sensor_h_mm:.6f} mm",
                f"full raster: {full_w or 'unknown'} x {full_h or 'unknown'} px",
                f"crop origin: ({crop_x}, {crop_y}) px",
                f"crop size: {crop_w or 'unknown'} x {crop_h or 'unknown'} px",
                f"active sensor: {active_w_mm:.9f} x {active_h_mm:.9f} mm",
                f"crop fraction: {_number(crop_w / full_w if full_w else 1.0, 9)} x {_number(crop_h / full_h if full_h else 1.0, 9)}",
                f"crop factor: {_number(full_w / crop_w if crop_w else 1.0, 9)} x {_number(full_h / crop_h if crop_h else 1.0, 9)}",
                f"origin convention: {region.get('origin', 'top-left')}",
            ],
            "accent": (90, 175, 175),
        },
        {
            "key": "exposure",
            "title": "EXPOSURE / SAMPLING",
            "lines": [
                f"time: {_number(exposure.get('time_s', 1.0 / 60.0), 9)} s",
                f"ISO: {_number(exposure.get('iso', 100.0), 3)}",
                f"sensor sweeps: {exposure.get('sensor_sweeps', 1)}",
                f"T5 pair budget: {exposure.get('t5_pair_budget', 20_000_000)}",
                f"rays emitted: {context.get('rays_emitted', 'see render summary')}",
                f"batches: {context.get('batches', 'see render summary')}",
            ],
            "accent": (215, 145, 70),
        },
        {
            "key": "output",
            "title": "COLOR / OUTPUT",
            "lines": [
                f"white level: {_number(profile.get('sensor_white_level'), 9)}",
                f"exposure compensation: {_number(profile.get('exposure_compensation_ev'), 6)} EV",
                f"white balance: {_vector(profile.get('white_balance'), 6)}",
                f"tone curve: {profile.get('tone_curve_mode') or 'not supplied'}",
                f"output space: {profile.get('output_space') or 'not supplied'}",
            ],
            "accent": (180, 180, 90),
        },
    ]
    groups = resolved_optics.get("groups", ())
    if isinstance(groups, (list, tuple)):
        group_sections = []
        for index, raw_group in enumerate(groups):
            group = _mapping(raw_group)
            group_sections.append({
                "key": f"lens_group_{index + 1}",
                "title": f"  OPTICAL GROUP {group.get('name', index + 1)}",
                "lines": [
                    f"center X: {_number(group.get('center_x_m'), 9)} m",
                    f"focal length: {_number(group.get('focal_length_mm'), 6)} mm",
                    f"clear radius: {_number(group.get('aperture_radius_mm'), 6)} mm",
                    f"thickness: {_number(group.get('thickness_mm'), 6)} mm",
                    f"front radius: {_number(group.get('radius_front_mm'), 6)} mm",
                    f"back radius: {_number(group.get('radius_back_mm'), 6)} mm",
                    f"index at reference: {_number(group.get('ior'), 9)}",
                    f"glass: {group.get('glass', lens_manifest.get('glass', 'not authored'))}",
                ],
                "accent": (85, 145, 190),
            })
        sections[3:3] = group_sections

    wavelengths = spectral_manifest.get("active_wavelengths_nm", ())
    wavelength_text = (
        ", ".join(f"{float(item):g}" for item in wavelengths)
        if isinstance(wavelengths, (list, tuple)) else "not resolved"
    )
    emitters = flash_manifest.get("emitters", ())
    emitter_lines = []
    if isinstance(emitters, (list, tuple)):
        for item in emitters:
            emitter = _mapping(item)
            emitter_lines.append(
                f"{emitter.get('key', 'emitter')}: {emitter.get('kind', 'unknown')} "
                f"scale={_number(emitter.get('emission_scale'), 6)} "
                f"CCT={_number(emitter.get('declared_cct_k'), 1)}K"
            )
    sections.insert(-2, {
        "key": "spectral_transport",
        "title": "SPECTRAL TRANSPORT",
        "lines": [
            f"authoring domain: {spectral_manifest.get('authoring_domain', 'not authored')}",
            f"transport: {spectral_manifest.get('transport_mode', 'not authored')}",
            f"lane semantics: {spectral_manifest.get('lane_semantics', 'legacy fixed spectral')}",
            f"transport ABI: v{spectral_manifest.get('transport_abi_version', 'unversioned')}",
            f"active bands: {spectral_manifest.get('active_band_count', 'unknown')} / capacity {spectral_manifest.get('material_band_capacity', 'unknown')}",
            f"wavelengths nm: {wavelength_text}",
            f"lens dispersion: {spectral_manifest.get('lens_dispersion_model', 'not authored')}",
            f"compatibility: {compatibility.get('spectral_grid', 'not resolved')}",
        ],
        "accent": (125, 105, 220),
    })
    sections.insert(-2, {
        "key": "flash",
        "title": "FLASH / EMITTER RIG",
        "lines": [
            f"asset key: {flash_manifest.get('asset_key', 'not authored')}",
            f"intensity scale: {_number(flash_manifest.get('intensity_scale'), 6)}",
            f"spectral model: {_mapping(flash_manifest.get('resolved_spectral_model')).get('kind', 'not resolved')}",
            f"modifier: {_mapping(flash_manifest.get('modifier')).get('mode', 'none')}",
            f"geometry compatibility: {compatibility.get('flash_geometry', 'not resolved')}",
            f"spectrum compatibility: {compatibility.get('flash_spectrum', 'not resolved')}",
            *emitter_lines,
        ],
        "accent": (225, 155, 75),
    })
    if pose_animations:
        animation = pose_animations[0]
        sections.insert(4, {
            "key": "pose_animations",
            "title": "POSE ANIMATIONS / FOCUS",
            "lines": [
                f"subset: {getattr(animation, 'name', 'pose animation')}",
                f"rack in: {int(getattr(animation, 'rack_frames', 0))} frames",
                f"hold: {int(getattr(animation, 'hold_frames', 0))} frames",
                f"rack out: {int(getattr(animation, 'rack_frames', 0))} frames",
                (
                    "focus locations: "
                    f"{int(getattr(animation, 'focus_location_count', 0))}"
                ),
                (
                    "replacement pose frame: "
                    f"{int(getattr(animation, 'replacement_frame_index', 0))}"
                ),
                f"animation key: {getattr(animation, 'animation_key', '')}",
            ],
            "accent": (170, 105, 205),
        })
    return sections


class CameraLoadoutBrowser:
    """Left-side expandable camera hierarchy using the repository widget."""

    GAUGE_H = 126

    def __init__(self, title: str = "CAMERA LOADOUT") -> None:
        from bass_viewer import ScrollableSubpanelList

        self.widget = ScrollableSubpanelList(
            title, max_height=220, key_prefix="camera_loadout"
        )
        self._signature: tuple[Any, ...] = ()
        self._surface = None
        self._font = None
        self.context: dict[str, Any] = {}
        self.progress_metrics: dict[str, Any] = {
            "convergence": 0.0,
            "convergence_velocity_per_pass": 0.0,
            "priority_share": 0.0,
            "working": False,
            "status": "waiting",
        }

    def sync(
        self,
        selected_payload: Mapping[str, Any] | None,
        *,
        fallback_order: Mapping[str, Any] | None = None,
        camera_profile: Any = None,
        progress_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        from bass_viewer import ModularSubpanelSpec

        context = selected_camera_context(
            selected_payload, fallback_order, camera_profile
        )
        if progress_metrics is not None:
            self.progress_metrics = dict(progress_metrics)
        sections = camera_loadout_sections(context)
        signature = tuple(
            (section["key"], section["title"], tuple(section["lines"]))
            for section in sections
        )
        self.context = context
        if signature == self._signature:
            return
        expanded = {
            spec.key: bool(spec.expanded) for spec in self.widget.subpanels
        }
        specs = [
            ModularSubpanelSpec(
                key=section["key"],
                title=section["title"],
                summary_lines=list(section["lines"]),
                expanded=expanded.get(
                    section["key"], section["key"] in {"selection", "lens"}
                ),
                accent_rgb=tuple(section["accent"]),
                payload={"section": section["key"], "context": context},
            )
            for section in sections
        ]
        self.widget.set_subpanels(specs)
        self._signature = signature

    @staticmethod
    def _fraction(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0

    def _draw_pie(
        self,
        surface: Any,
        center: tuple[int, int],
        radius: int,
        fraction: float,
        color: tuple[int, int, int],
    ) -> None:
        import pygame

        fraction = self._fraction(fraction)
        pygame.draw.circle(surface, (45, 49, 58), center, radius)
        if fraction > 0.0:
            steps = max(3, int(math.ceil(40 * fraction)))
            points = [center]
            for index in range(steps + 1):
                angle = -math.pi / 2.0 + 2.0 * math.pi * fraction * index / steps
                points.append((
                    int(round(center[0] + math.cos(angle) * radius)),
                    int(round(center[1] + math.sin(angle) * radius)),
                ))
            pygame.draw.polygon(surface, color, points)
        pygame.draw.circle(surface, (125, 132, 145), center, radius, 1)
        percent = self._font.render(
            f"{fraction * 100:3.0f}%", True, (235, 238, 242)
        )
        surface.blit(
            percent,
            (
                center[0] - percent.get_width() // 2,
                center[1] - percent.get_height() // 2,
            ),
        )

    def _render_progress_header(self, surface: Any, width: int) -> None:
        import pygame

        convergence = self._fraction(
            self.progress_metrics.get("convergence", 0.0)
        )
        priority = self._fraction(
            self.progress_metrics.get("priority_share", 0.0)
        )
        working = bool(self.progress_metrics.get("working", False))
        status = str(self.progress_metrics.get("status", "waiting")).upper()
        try:
            convergence_velocity = float(
                self.progress_metrics.get(
                    "convergence_velocity_per_pass", 0.0
                )
            )
        except (TypeError, ValueError):
            convergence_velocity = 0.0
        if not math.isfinite(convergence_velocity):
            convergence_velocity = 0.0
        radius = max(18, min(29, (width - 34) // 5))
        left_x, right_x = width // 4, width * 3 // 4
        center_y = 38
        self._draw_pie(
            surface, (left_x, center_y), radius, convergence, (64, 178, 135)
        )
        self._draw_pie(
            surface, (right_x, center_y), radius, priority, (92, 139, 224)
        )
        for center_x, label in (
            (left_x, "CONVERGENCE"),
            (right_x, "PRIORITY SHARE"),
        ):
            text = self._font.render(label, True, (176, 182, 194))
            surface.blit(
                text,
                (center_x - text.get_width() // 2, center_y + radius + 4),
            )
        status_color = (244, 169, 58) if working else (137, 145, 158)
        status_text = f"● {status}" if working else status
        status_surface = self._font.render(status_text, True, status_color)
        surface.blit(
            status_surface,
            (
                width // 2 - status_surface.get_width() // 2,
                self.GAUGE_H - 2 * status_surface.get_height() - 5,
            ),
        )
        velocity_surface = self._font.render(
            (
                "dC/dpass "
                f"{convergence_velocity:+.3e} pass^-1"
            ),
            True,
            (164, 204, 190),
        )
        surface.blit(
            velocity_surface,
            (
                width // 2 - velocity_surface.get_width() // 2,
                self.GAUGE_H - velocity_surface.get_height() - 3,
            ),
        )
        pygame.draw.line(
            surface,
            (60, 65, 76),
            (8, self.GAUGE_H - 1),
            (width - 8, self.GAUGE_H - 1),
        )

    def render(self, destination: Any, rect: tuple[int, int, int, int]) -> None:
        import pygame

        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 12)
            self.widget.TITLE_H = 28
            self.widget.HEADER_H = 28
        if self._surface is None or self._surface.get_size() != (width, height):
            self._surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._surface.fill((18, 20, 26, 245))
        gauge_height = min(self.GAUGE_H, max(0, height - 64))
        if gauge_height == self.GAUGE_H:
            self._render_progress_header(self._surface, width)
        else:
            gauge_height = 0
        self.widget.max_height = max(
            60, height - gauge_height - self.widget.TITLE_H - 4
        )
        self.widget.render(self._surface, self._font, 0, gauge_height, width)
        destination.blit(self._surface, (x, y))

    def handle_event(
        self, event: Any, rect: tuple[int, int, int, int]
    ) -> str | None:
        import pygame

        x, y, width, height = map(int, rect)
        host = pygame.Rect(x, y, width, height)
        if event.type == pygame.MOUSEWHEEL:
            pointer = getattr(event, "pos", pygame.mouse.get_pos())
            if not host.collidepoint(pointer):
                return None
            return "scroll" if self.widget.handle_scroll(-int(event.y)) else ""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not host.collidepoint(event.pos):
                return None
            return self.widget.handle_click(
                int(event.pos[0]) - x, int(event.pos[1]) - y
            )
        return None


__all__ = [
    "CameraLoadoutBrowser",
    "camera_loadout_sections",
    "selected_camera_context",
]
