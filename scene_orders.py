"""Declarative scene orders for the spectral thick-lens exposure pipeline.

The compiler intentionally owns subject geometry only.  It retains the proven
sensor, aperture, lens, and flash geometry from a supplied ``TracerScene`` and
replaces that scene's ``object`` triangle group with order-authored planes and
font-outline extrusions.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompiledOrderReport:
    job_id: str
    token: str
    plane_triangles: int
    glyph_triangles: int
    retained_base_triangles: int
    total_triangles: int
    material_names: tuple[str, ...]
    flash_scale: float
    removed_subject_emitters: int = 0
    camera_position_m: tuple[float, float, float] | None = None
    camera_target_m: tuple[float, float, float] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_order(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    validate_order(payload)
    payload["_source_path"] = os.path.abspath(path)
    return payload


def validate_order(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("scene order root must be an object")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"scene order schema_version must be {SCHEMA_VERSION}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("scene order requires a non-empty jobs array")
    seen: set[str] = set()
    for i, raw in enumerate(jobs):
        if not isinstance(raw, dict):
            raise ValueError(f"jobs[{i}] must be an object")
        job_id = str(raw.get("id", "")).strip()
        if not job_id or job_id in seen:
            raise ValueError(f"jobs[{i}].id must be non-empty and unique")
        seen.add(job_id)
        token = str(raw.get("token", ""))
        if not token:
            raise ValueError(f"job {job_id!r} requires a token")


def resolved_jobs(payload: dict[str, Any], job_id: str | None = None) -> list[dict[str, Any]]:
    validate_order(payload)
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    jobs = [_deep_merge(defaults, raw) for raw in payload["jobs"]]
    if job_id is not None:
        jobs = [job for job in jobs if str(job.get("id")) == str(job_id)]
        if not jobs:
            raise ValueError(f"scene order has no job {job_id!r}")
    for job in jobs:
        _validate_resolved_job(job)
    return jobs


def _vec(value: Any, name: str, n: int = 3) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != n or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain {n} finite numbers")
    return arr


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    x = float(value)
    if not math.isfinite(x) or (x < 0.0 if allow_zero else x <= 0.0):
        op = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {op}")
    return x


def _validate_resolved_job(job: dict[str, Any]) -> None:
    supported = {
        "id", "token", "image", "camera", "exposure", "flash", "font",
        "planes", "materials", "geometry"
    }
    unknown = sorted(set(job) - supported)
    if unknown:
        raise ValueError(f"job {job['id']!r}: unsupported fields {unknown}")
    token = str(job.get("token", ""))
    if not token.strip():
        raise ValueError(f"job {job['id']!r}: token must be non-empty text")
    contracts = {
        "image": {"width", "height", "region"},
        "camera": {
            "focal_mm", "aperture_mm", "position_m", "target_m", "focus_target_m", "up"
        },
        "exposure": {"time_s", "iso", "sensor_sweeps", "t5_pair_budget"},
        "flash": {"intensity_scale"},
        "font": {"family", "weight", "style", "file"},
    }
    for section, allowed in contracts.items():
        value = job.get(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"job {job['id']!r}: {section} must be an object")
        extra = sorted(set(value) - allowed)
        if extra:
            raise ValueError(
                f"job {job['id']!r}: unsupported {section} fields {extra}; "
                "refusing to silently ignore requested behavior"
            )
    image = job.get("image", {})
    full_width = int(image.get("width", 64))
    full_height = int(image.get("height", 64))
    if full_width <= 0 or full_height <= 0:
        raise ValueError("image width and height must be positive")
    region = image.get("region", {})
    if not isinstance(region, dict):
        raise ValueError("image.region must be an object")
    region_extra = sorted(set(region) - {"x", "y", "width", "height"})
    if region_extra:
        raise ValueError(f"image.region has unsupported fields {region_extra}")
    region_x = int(region.get("x", 0))
    region_y = int(region.get("y", 0))
    region_width = int(region.get("width", full_width))
    region_height = int(region.get("height", full_height))
    if region_x < 0 or region_y < 0 or region_width <= 0 or region_height <= 0:
        raise ValueError("image.region requires non-negative x/y and positive width/height")
    if region_x + region_width > full_width or region_y + region_height > full_height:
        raise ValueError("image.region must fit within image width and height")
    camera = job.get("camera", {})
    _positive(job.get("camera", {}).get("focal_mm", 35.0), "camera.focal_mm")
    _positive(job.get("camera", {}).get("aperture_mm", 25.0), "camera.aperture_mm")
    has_position = "position_m" in camera
    has_target = "target_m" in camera
    if has_position != has_target:
        raise ValueError("camera.position_m and camera.target_m must be supplied together")
    if has_position:
        position = _vec(camera["position_m"], "camera.position_m")
        target = _vec(camera["target_m"], "camera.target_m")
        focus_target = _vec(camera.get("focus_target_m", target), "camera.focus_target_m")
        up = _vec(camera.get("up", [0, 0, 1]), "camera.up")
        fwd = target - position
        if float(np.linalg.norm(fwd)) <= 1.0e-10:
            raise ValueError("camera.position_m and camera.target_m must be distinct")
        if float(np.linalg.norm(np.cross(fwd, up))) <= 1.0e-10:
            raise ValueError("camera.up must not be parallel to the viewing axis")
        if float(np.dot(focus_target - position, fwd)) <= 0.0:
            raise ValueError("camera.focus_target_m must be in front of the camera")
    _positive(job.get("exposure", {}).get("time_s", 1.0/60.0), "exposure.time_s")
    _positive(job.get("exposure", {}).get("iso", 100.0), "exposure.iso")
    _positive(job.get("exposure", {}).get("sensor_sweeps", 1), "exposure.sensor_sweeps")
    _positive(job.get("flash", {}).get("intensity_scale", 1.0), "flash.intensity_scale")
    geometry = job.get("geometry", {})
    planes = job.get("planes", [])
    if not isinstance(geometry, dict):
        raise ValueError(f"job {job['id']!r}: geometry must be an object")
    if not isinstance(planes, list) or not planes:
        raise ValueError(f"job {job['id']!r}: at least one plane is required")
    plane_ids = set()
    for i, plane in enumerate(planes):
        if not isinstance(plane, dict):
            raise ValueError(f"job {job['id']!r}: planes[{i}] must be an object")
        pid = str(plane.get("id", "")).strip()
        if not pid or pid in plane_ids:
            raise ValueError(f"job {job['id']!r}: plane ids must be unique")
        plane_ids.add(pid)
        plane_extra = sorted(set(plane) - {
            "id", "center_m", "normal", "up", "size_m", "thickness_m", "material"
        })
        if plane_extra:
            raise ValueError(f"plane {pid}: unsupported fields {plane_extra}")
        _vec(plane.get("center_m", [0, 0, 0]), f"plane {pid}.center_m")
        _vec(plane.get("normal", [1, 0, 0]), f"plane {pid}.normal")
        _vec(plane.get("up", [0, 0, 1]), f"plane {pid}.up")
        size = _vec(plane.get("size_m", [0.5, 0.5]), f"plane {pid}.size_m", 2)
        if np.any(size <= 0.0):
            raise ValueError(f"plane {pid}.size_m must be positive")
        _positive(plane.get("thickness_m", 0.01), f"plane {pid}.thickness_m")
    target = str(geometry.get("embed_plane", planes[0]["id"]))
    if target not in plane_ids:
        raise ValueError(f"job {job['id']!r}: embed_plane {target!r} is not declared")
    _positive(geometry.get("height_m", 0.2), "geometry.height_m")
    if "extrusion_depth_ratio" in geometry:
        _positive(geometry["extrusion_depth_ratio"],
                  "geometry.extrusion_depth_ratio")
        if "depth_m" in geometry:
            raise ValueError(
                "geometry.depth_m and geometry.extrusion_depth_ratio are mutually exclusive"
            )
    else:
        _positive(geometry.get("depth_m", 0.03), "geometry.depth_m")
    embed = float(geometry.get("embed_fraction", 0.5))
    if not 0.0 <= embed <= 1.0:
        raise ValueError("geometry.embed_fraction must be in [0,1]")
    profile = str(geometry.get("profile", "straight"))
    geometry_extra = sorted(set(geometry) - {
        "embed_plane", "height_m", "depth_m", "extrusion_depth_ratio",
        "embed_fraction", "offset_m",
        "profile", "profile_segments", "profile_bulge", "outline_subdivisions",
        "cap_grid", "material", "text_box_m", "line_height_m", "line_spacing",
        "horizontal_align", "vertical_align"
    })
    if geometry_extra:
        raise ValueError(f"geometry has unsupported fields {geometry_extra}")
    if profile not in ("straight", "circular"):
        raise ValueError("geometry.profile must be 'straight' or 'circular'")
    if int(geometry.get("outline_subdivisions", 2)) < 1:
        raise ValueError("geometry.outline_subdivisions must be positive")
    if int(geometry.get("cap_grid", 28)) < 4:
        raise ValueError("geometry.cap_grid must be at least 4")
    if profile == "circular" and int(geometry.get("profile_segments", 8)) < 2:
        raise ValueError("circular geometry.profile_segments must be at least 2")
    if "text_box_m" in geometry:
        text_box = _vec(geometry["text_box_m"], "geometry.text_box_m", 2)
        if np.any(text_box <= 0.0):
            raise ValueError("geometry.text_box_m must be positive")
        _positive(geometry.get("line_height_m", geometry.get("height_m", 0.2)),
                  "geometry.line_height_m")
        _positive(geometry.get("line_spacing", 1.2), "geometry.line_spacing")
        if str(geometry.get("horizontal_align", "center")) != "center":
            raise ValueError("only centered geometry.horizontal_align is currently supported")
        if str(geometry.get("vertical_align", "center")) != "center":
            raise ValueError("only centered geometry.vertical_align is currently supported")


def order_runtime_settings(job: dict[str, Any]) -> dict[str, Any]:
    """Return consumed camera/exposure/output settings for a resolved job."""
    image = job.get("image", {})
    camera = job.get("camera", {})
    exposure = job.get("exposure", {})
    full_width = int(image.get("width", 64))
    full_height = int(image.get("height", 64))
    region = image.get("region", {})
    region_x = int(region.get("x", 0))
    region_y = int(region.get("y", 0))
    region_width = int(region.get("width", full_width))
    region_height = int(region.get("height", full_height))
    return {
        "width": region_width,
        "height": region_height,
        "full_width": full_width,
        "full_height": full_height,
        "region": {
            "x": region_x, "y": region_y,
            "width": region_width, "height": region_height,
        },
        "focal_mm": float(camera.get("focal_mm", 35.0)),
        "aperture_mm": float(camera.get("aperture_mm", 25.0)),
        "iso": float(exposure.get("iso", 100.0)),
        "exposure_time_s": float(exposure.get("time_s", 1.0 / 60.0)),
        "sensor_sweeps": int(exposure.get("sensor_sweeps", 1)),
        "pair_budget": int(exposure.get("t5_pair_budget", 20_000_000)),
    }


def camera_pose(job: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Return the authored camera frame, or None for the canonical lab pose."""
    camera = job.get("camera", {})
    if "position_m" not in camera:
        return None
    position = _vec(camera["position_m"], "camera.position_m")
    target = _vec(camera["target_m"], "camera.target_m")
    fwd = target - position
    fwd /= float(np.linalg.norm(fwd))
    up = _vec(camera.get("up", [0, 0, 1]), "camera.up")
    up -= fwd * float(np.dot(up, fwd))
    up /= float(np.linalg.norm(up))
    right = np.cross(fwd, up)
    right /= float(np.linalg.norm(right))
    up = np.cross(right, fwd)
    return {"position": position, "target": target, "fwd": fwd, "right": right, "up": up}


def camera_focus_distance(job: dict[str, Any]) -> float | None:
    """Return the axial sensor-to-focus-plane distance in authored world space."""
    pose = camera_pose(job)
    if pose is None:
        return None
    focus_target = _vec(
        job.get("camera", {}).get("focus_target_m", pose["target"]),
        "camera.focus_target_m",
    )
    distance = float(np.dot(focus_target - pose["position"], pose["fwd"]))
    if distance <= 0.0:
        raise ValueError("camera.focus_target_m must be in front of the camera")
    return distance


def sensor_tile(job: dict[str, Any], sensor_w_m: float, sensor_h_m: float) -> dict[str, Any]:
    """Map the requested top-left image ROI onto the physical sensor plane."""
    runtime = order_runtime_settings(job)
    region = runtime["region"]
    full_width = runtime["full_width"]
    full_height = runtime["full_height"]
    width_fraction = float(region["width"]) / float(full_width)
    height_fraction = float(region["height"]) / float(full_height)
    center_x = (float(region["x"]) + 0.5 * float(region["width"])) / float(full_width)
    center_y = (float(region["y"]) + 0.5 * float(region["height"])) / float(full_height)
    return {
        "sensor_w_m": float(sensor_w_m) * width_fraction,
        "sensor_h_m": float(sensor_h_m) * height_fraction,
        "right_offset_m": (center_x - 0.5) * float(sensor_w_m),
        "up_offset_m": (0.5 - center_y) * float(sensor_h_m),
        "full_width": int(full_width),
        "full_height": int(full_height),
        "region": dict(region),
        "origin": "top-left",
    }


def composition_metadata(job: dict[str, Any]) -> dict[str, Any]:
    runtime = order_runtime_settings(job)
    pose = camera_pose(job)
    return {
        "coordinate_space": "full_sensor_pixels",
        "origin": "top-left",
        "full_frame": {"width": runtime["full_width"], "height": runtime["full_height"]},
        "region": dict(runtime["region"]),
        "camera": None if pose is None else {
            "position_m": pose["position"].tolist(),
            "target_m": pose["target"].tolist(),
            "up": pose["up"].tolist(),
            "focus_target_m": _vec(
                job.get("camera", {}).get("focus_target_m", pose["target"]),
                "camera.focus_target_m",
            ).tolist(),
            "focus_distance_m": camera_focus_distance(job),
        },
        "render_basis": "camera_local_canonical_x",
    }


def _world_to_canonical_camera(
    points: np.ndarray,
    pose: dict[str, np.ndarray] | None,
    canonical_origin: np.ndarray,
) -> np.ndarray:
    if pose is None:
        return np.asarray(points, np.float64)
    delta = np.asarray(points, np.float64) - pose["position"]
    return np.stack((
        canonical_origin[0] - delta @ pose["fwd"],
        canonical_origin[1] + delta @ pose["right"],
        canonical_origin[2] + delta @ pose["up"],
    ), axis=-1)


def _plane_basis(plane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = _vec(plane.get("center_m", [0, 0, 0]), "plane.center_m")
    normal = _vec(plane.get("normal", [1, 0, 0]), "plane.normal")
    normal /= max(float(np.linalg.norm(normal)), 1.0e-15)
    up = _vec(plane.get("up", [0, 0, 1]), "plane.up")
    up -= normal * float(np.dot(up, normal))
    if float(np.linalg.norm(up)) < 1.0e-10:
        raise ValueError("plane.up must not be parallel to plane.normal")
    up /= float(np.linalg.norm(up))
    right = np.cross(up, normal)
    right /= max(float(np.linalg.norm(right)), 1.0e-15)
    return center, normal, right, up


def _box_plane_triangles(plane: dict[str, Any]) -> np.ndarray:
    center, normal, right, up = _plane_basis(plane)
    width, height = _vec(plane.get("size_m", [0.5, 0.5]), "plane.size_m", 2)
    thickness = float(plane.get("thickness_m", 0.01))
    front = center
    back = center - normal * thickness
    corners = []
    for c in (front, back):
        corners.append([
            c - right * width/2 - up * height/2,
            c + right * width/2 - up * height/2,
            c + right * width/2 + up * height/2,
            c - right * width/2 + up * height/2,
        ])
    f, b = corners
    faces = [
        (f[0], f[1], f[2]), (f[0], f[2], f[3]),
        (b[0], b[2], b[1]), (b[0], b[3], b[2]),
    ]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        faces.extend(((f[i], b[i], b[j]), (f[i], b[j], f[j])))
    return np.asarray(faces, np.float64)


def _points_inside_even_odd(points: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    inside = np.zeros(pts.shape[0], dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    for poly in contours:
        p = np.asarray(poly, np.float64)
        x0, y0 = p[:-1, 0], p[:-1, 1]
        x1, y1 = p[1:, 0], p[1:, 1]
        hit = np.zeros(pts.shape[0], dtype=bool)
        for i in range(x0.size):
            crosses = ((y0[i] > y) != (y1[i] > y))
            x_at_y = (x1[i] - x0[i]) * (y - y0[i]) / (y1[i] - y0[i] + 1.0e-300) + x0[i]
            hit ^= crosses & (x < x_at_y)
        inside ^= hit
    return inside


def _font_contours(token: str, font: dict[str, Any], curve_steps: int) -> list[np.ndarray]:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    kwargs: dict[str, Any] = {
        "family": str(font.get("family", "DejaVu Sans")),
        "weight": str(font.get("weight", "bold")),
        "style": str(font.get("style", "normal")),
    }
    if font.get("file"):
        kwargs["fname"] = os.path.abspath(str(font["file"]))
    path = TextPath((0.0, 0.0), token, size=1.0, prop=FontProperties(**kwargs))
    polys = path.to_polygons(closed_only=True)
    contours: list[np.ndarray] = []
    for raw in polys:
        p = np.asarray(raw, np.float64)
        if p.shape[0] < 3:
            continue
        if np.linalg.norm(p[0] - p[-1]) > 1.0e-12:
            p = np.vstack([p, p[0]])
        # Resample long outline edges so side-wall curvature and cap constraints
        # remain stable across fonts with sparse straight segments.
        q: list[np.ndarray] = []
        for a, b in zip(p[:-1], p[1:]):
            q.append(a)
            for k in range(1, max(1, curve_steps)):
                q.append(a + (b - a) * (k / max(1, curve_steps)))
        q.append(p[0])
        contours.append(np.asarray(q, np.float64))
    if not contours:
        raise ValueError(f"font produced no closed outline for token {token!r}")
    return contours


def _normalize_contours(contours: list[np.ndarray], height_m: float) -> list[np.ndarray]:
    all_pts = np.concatenate([p[:-1] for p in contours], axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    if hi[1] - lo[1] <= 1.0e-12:
        raise ValueError("glyph outline has zero height")
    scale = height_m / (hi[1] - lo[1])
    center = 0.5 * (lo + hi)
    return [(p - center) * scale for p in contours]


def _font_key(font: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(font.get("family", "DejaVu Sans")),
        str(font.get("weight", "bold")),
        str(font.get("style", "normal")),
        os.path.abspath(str(font["file"])) if font.get("file") else "",
    )


@functools.lru_cache(maxsize=4096)
def _cached_font_path(token: str, key: tuple[str, str, str, str]):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    family, weight, style, filename = key
    kwargs: dict[str, Any] = {"family": family, "weight": weight, "style": style}
    if filename:
        kwargs["fname"] = filename
    return TextPath((0.0, 0.0), token, size=1.0, prop=FontProperties(**kwargs))


def _font_path(token: str, font: dict[str, Any]):
    return _cached_font_path(str(token), _font_key(font))


@functools.lru_cache(maxsize=8192)
def _cached_text_advance(token: str, key: tuple[str, str, str, str]) -> float:
    return float(_cached_font_path(token, key).get_extents().width) if token else 0.0


def _text_advance(token: str, font: dict[str, Any]) -> float:
    return _cached_text_advance(str(token), _font_key(font))


def _wrap_paragraph(text: str, font: dict[str, Any], max_advance: float) -> list[str]:
    """Greedy font-metric wrapping with explicit newlines and hard word breaks."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if _text_advance(candidate, font) <= max_advance:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            while word and _text_advance(word, font) > max_advance:
                split_at = 1
                for index in range(2, len(word) + 1):
                    if _text_advance(word[:index], font) > max_advance:
                        break
                    split_at = index
                lines.append(word[:split_at])
                word = word[split_at:]
            current = word
        if current:
            lines.append(current)
    return lines or [""]


def layout_paragraph(job: dict[str, Any]) -> dict[str, Any]:
    """Resolve wrapped lines and their shared physical scale inside a text box."""
    geometry = job.get("geometry", {})
    text_box = _vec(geometry["text_box_m"], "geometry.text_box_m", 2)
    font = job.get("font", {})
    reference_height = max(float(_font_path("Hg", font).get_extents().height), 1.0e-12)
    max_line_height = float(geometry.get("line_height_m", geometry.get("height_m", 0.2)))
    line_spacing = float(geometry.get("line_spacing", 1.2))

    def resolve(line_height: float) -> tuple[list[str], float]:
        scale = line_height / reference_height
        lines = _wrap_paragraph(str(job["token"]), font, float(text_box[0]) / scale)
        block_height = line_height * (1.0 + line_spacing * max(0, len(lines) - 1))
        return lines, block_height

    lines, block_height = resolve(max_line_height)
    line_height = max_line_height
    if block_height > float(text_box[1]):
        low = max_line_height * 1.0e-4
        high = max_line_height
        for _ in range(32):
            candidate = 0.5 * (low + high)
            candidate_lines, candidate_height = resolve(candidate)
            if candidate_height <= float(text_box[1]):
                low = candidate
                lines, block_height = candidate_lines, candidate_height
            else:
                high = candidate
        line_height = low
        lines, block_height = resolve(line_height)
    return {
        "lines": lines,
        "line_height_m": float(line_height),
        "line_spacing": line_spacing,
        "block_height_m": float(block_height),
        "text_box_m": text_box,
    }


def resolved_glyph_height(job: dict[str, Any]) -> float:
    """Physical glyph height after paragraph formatting has chosen its scale."""

    geometry = job.get("geometry", {})
    if "text_box_m" in geometry:
        return float(layout_paragraph(job)["line_height_m"])
    return float(geometry.get("height_m", 0.2))


def resolved_glyph_depth(job: dict[str, Any]) -> float:
    """Extrusion depth derived from final glyph scale, or legacy fixed depth."""

    geometry = job.get("geometry", {})
    if "extrusion_depth_ratio" in geometry:
        return resolved_glyph_height(job) * float(geometry["extrusion_depth_ratio"])
    return float(geometry.get("depth_m", 0.03))


def _paragraph_contours(job: dict[str, Any]) -> list[np.ndarray]:
    layout = layout_paragraph(job)
    font = job.get("font", {})
    curve_steps = int(job.get("geometry", {}).get("outline_subdivisions", 2))
    reference_height = max(float(_font_path("Hg", font).get_extents().height), 1.0e-12)
    scale = float(layout["line_height_m"]) / reference_height
    lines = list(layout["lines"])
    pitch = float(layout["line_height_m"]) * float(layout["line_spacing"])
    block_center = 0.5 * float(max(0, len(lines) - 1)) * pitch
    contours: list[np.ndarray] = []
    for row, line in enumerate(lines):
        if not line:
            continue
        raw = _font_contours(line, font, curve_steps)
        points = np.concatenate([contour[:-1] for contour in raw], axis=0)
        center_x = 0.5 * float(points[:, 0].min() + points[:, 0].max())
        center_y = 0.5 * float(points[:, 1].min() + points[:, 1].max())
        line_y = block_center - float(row) * pitch
        for contour in raw:
            shifted = contour.copy()
            shifted[:, 0] = (shifted[:, 0] - center_x) * scale
            shifted[:, 1] = (shifted[:, 1] - center_y) * scale + line_y
            contours.append(shifted)
    if not contours:
        raise ValueError("paragraph produced no visible glyph contours")
    all_points = np.concatenate([contour[:-1] for contour in contours], axis=0)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    extent = np.maximum(hi - lo, 1.0e-15)
    text_box = np.asarray(layout["text_box_m"], np.float64)
    fit_scale = min(1.0, float(np.min(text_box / extent)))
    bounds_center = 0.5 * (lo + hi)
    contours = [(contour - bounds_center) * fit_scale for contour in contours]
    return contours


def _triangulate_caps(contours: list[np.ndarray], grid_n: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import Delaunay

    boundary = np.concatenate([p[:-1] for p in contours], axis=0)
    lo, hi = boundary.min(axis=0), boundary.max(axis=0)
    gx = np.linspace(lo[0], hi[0], max(4, int(grid_n)))
    gy = np.linspace(lo[1], hi[1], max(4, int(grid_n)))
    grid = np.stack(np.meshgrid(gx, gy, indexing="xy"), axis=-1).reshape(-1, 2)
    grid = grid[_points_inside_even_odd(grid, contours)]
    points = np.unique(np.round(np.vstack([boundary, grid]), decimals=12), axis=0)
    delaunay = Delaunay(points)
    tri = np.asarray(delaunay.simplices, np.int32)
    p = points[tri]
    samples = np.stack([
        p.mean(axis=1),
        0.60*p[:, 0] + 0.20*p[:, 1] + 0.20*p[:, 2],
        0.20*p[:, 0] + 0.60*p[:, 1] + 0.20*p[:, 2],
        0.20*p[:, 0] + 0.20*p[:, 1] + 0.60*p[:, 2],
    ], axis=1)
    keep = _points_inside_even_odd(samples.reshape(-1, 2), contours).reshape(-1, 4).all(axis=1)
    area2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    # Match the compiled-scene normal contract. Dense font outlines can make
    # Delaunay return almost-collinear slivers that are numerically nonzero in
    # 2-D but collapse after world transformation.
    keep &= np.abs(area2) > 1.0e-14
    return points, tri[keep]


def _sanitize_scaled_contours(
    contours: list[np.ndarray], resolved_height: float
) -> list[np.ndarray]:
    """Remove duplicate microscopic outline steps before making sidewalls."""

    tolerance = max(1.0e-12, float(resolved_height) * 1.0e-9)
    cleaned: list[np.ndarray] = []
    for contour in contours:
        points = np.asarray(contour, np.float64).reshape(-1, 2)
        if points.shape[0] > 1 and np.linalg.norm(points[-1] - points[0]) <= tolerance:
            points = points[:-1]
        kept: list[np.ndarray] = []
        for point in points:
            if not kept or np.linalg.norm(point - kept[-1]) > tolerance:
                kept.append(point)
        if len(kept) > 1 and np.linalg.norm(kept[-1] - kept[0]) <= tolerance:
            kept.pop()
        if len(kept) < 3:
            raise ValueError("glyph contour collapsed below three distinct points")
        closed = np.vstack([np.asarray(kept, np.float64), kept[0]])
        cleaned.append(np.ascontiguousarray(closed, dtype=np.float64))
    return cleaned


def _glyph_triangles(job: dict[str, Any], plane: dict[str, Any]) -> np.ndarray:
    geometry = job.get("geometry", {})
    contours = (
        _paragraph_contours(job)
        if "text_box_m" in geometry
        else _normalize_contours(
            _font_contours(str(job["token"]), job.get("font", {}),
                           int(geometry.get("outline_subdivisions", 2))),
            float(geometry.get("height_m", 0.2)),
        )
    )
    contours = _sanitize_scaled_contours(contours, resolved_glyph_height(job))
    cap_points, cap_tri = _triangulate_caps(contours, int(geometry.get("cap_grid", 28)))
    center, normal, right, up = _plane_basis(plane)
    offset = _vec(geometry.get("offset_m", [0.0, 0.0]), "geometry.offset_m", 2)
    center = center + right * offset[0] + up * offset[1]
    depth = resolved_glyph_depth(job)
    embed = float(geometry.get("embed_fraction", 0.5))
    front_d, back_d = depth * (1.0 - embed), -depth * embed
    profile = str(geometry.get("profile", "straight"))
    segments = int(geometry.get("profile_segments", 1 if profile == "straight" else 8))
    segments = max(1, segments)
    bulge = float(geometry.get("profile_bulge", 0.06 if profile == "circular" else 0.0))

    def world(local: np.ndarray, d: float, scale: float = 1.0) -> np.ndarray:
        return (center[None, :] + normal[None, :] * d
                + right[None, :] * (local[:, 0:1] * scale)
                + up[None, :] * (local[:, 1:2] * scale))

    tris: list[np.ndarray] = []
    # Front/back caps.  Orient explicitly relative to the plane normal.
    front_pts = world(cap_points, front_d)
    back_pts = world(cap_points, back_d)
    for ids in cap_tri:
        a, b, c = ids.tolist()
        local_area = np.cross(
            np.array([0.0, *(cap_points[b] - cap_points[a])]),
            np.array([0.0, *(cap_points[c] - cap_points[a])]),
        )[0]
        front_ids = (a, b, c) if local_area >= 0.0 else (a, c, b)
        back_ids = tuple(reversed(front_ids))
        tris.append(front_pts[list(front_ids)])
        tris.append(back_pts[list(back_ids)])

    depths = np.linspace(back_d, front_d, segments + 1)
    scales = np.ones_like(depths)
    if profile == "circular":
        scales += bulge * np.sin(np.linspace(0.0, math.pi, segments + 1))
    for contour in contours:
        local = contour[:-1]
        rings = [world(local, float(d), float(s)) for d, s in zip(depths, scales)]
        for r0, r1 in zip(rings[:-1], rings[1:]):
            for i in range(local.shape[0]):
                j = (i + 1) % local.shape[0]
                tris.append(np.asarray([r0[i], r0[j], r1[j]]))
                tris.append(np.asarray([r0[i], r1[j], r1[i]]))
    return np.ascontiguousarray(np.asarray(tris, np.float64).reshape(-1, 3, 3))


def _triangle_normals(tri: np.ndarray) -> np.ndarray:
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mag = np.linalg.norm(cr, axis=1, keepdims=True)
    if np.any(mag <= 1.0e-14):
        raise ValueError("compiled scene contains degenerate triangles")
    return np.ascontiguousarray(cr / mag, dtype=np.float64)


def _display_rgb_weights(wl_nm: np.ndarray) -> np.ndarray:
    """Per-wavelength display RGB weights, replicating the native renderer's
    ``band_to_display_rgb`` (csrc/kernels/ray_tracer.cpp) exactly.  Authored
    materials must be baked against the SAME transfer the sensor path uses to
    turn per-band energy into display RGB, or authored colours drift."""
    out = np.zeros((wl_nm.size, 3), np.float64)
    for i, raw in enumerate(np.asarray(wl_nm, np.float64)):
        wl = min(700.0, max(380.0, float(raw)))
        r = g = b = 0.0
        if wl < 440.0:
            r = -(wl - 440.0) / 60.0
            b = 1.0
        elif wl < 490.0:
            g = (wl - 440.0) / 50.0
            b = 1.0
        elif wl < 510.0:
            g = 1.0
            b = -(wl - 510.0) / 20.0
        elif wl < 580.0:
            r = (wl - 510.0) / 70.0
            g = 1.0
        elif wl < 645.0:
            r = 1.0
            g = -(wl - 645.0) / 65.0
        else:
            r = 1.0
        edge = 1.0
        if wl < 420.0:
            edge = 0.3 + 0.7 * (wl - 380.0) / 40.0
        elif wl > 645.0:
            edge = 0.3 + 0.7 * (700.0 - wl) / 55.0
        out[i] = (r * edge, g * edge, b * edge)
    return out


def _material_payload(authored: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "plane":
        defaults = {
            "albedo_rgb": [0.003, 0.003, 0.003], "reflectivity": 0.015,
            "diffusion": 0.92, "absorption": 0.965, "roughness": 0.88,
            "metallic": 0.0, "transmission": 0.0, "ior": 1.5,
            "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        }
    else:
        defaults = {
            "albedo_rgb": [0.72, 0.008, 0.006], "reflectivity": 0.48,
            "diffusion": 0.20, "absorption": 0.32, "roughness": 0.075,
            "metallic": 0.08, "transmission": 0.0, "ior": 1.52,
            "opacity": 1.0, "emission_rgb": [0.0, 0.0, 0.0],
        }
    payload = _deep_merge(defaults, authored)
    # Native BDPT consumes explicit SpectralBandRecord rows.  PBR/albedo fields
    # alone only populate compatibility records and otherwise yield a black
    # spectral surface.  Bake the authored linear RGB into the renderer's eight
    # visible sample wavelengths.
    #
    # Colorimetric contract: the display path multiplies each band's energy by
    # band_to_display_rgb weights, so the baked curve must satisfy
    # W.T @ curve ∝ authored rgb.  Solve the 8-band curve directly as a
    # non-negative least-squares problem against the exact display transfer,
    # with a mild second-difference smoothness prior so the metamer stays a
    # physically plausible smooth reflectance rather than isolated spikes.
    # (A 700 nm-centred Gaussian red basis — the previous scheme — rendered
    # authored reds dim and yellow-shifted: the display transfer edge-attenuates
    # 700 nm to 0.3 and its 614 nm tail carries green weight 0.477.)
    c_light = 299_792_458.0
    wavelengths_nm = np.linspace(700.0, 400.0, 8)
    rgb = np.clip(np.asarray(payload.get("albedo_rgb", [0.5, 0.5, 0.5]), np.float64), 0.0, 1.0)
    weights = _display_rgb_weights(wavelengths_nm)          # (8, 3)
    from scipy.optimize import nnls
    second_diff = (np.eye(8, k=0) * -2.0 + np.eye(8, k=1) + np.eye(8, k=-1))[1:-1]
    smooth = 0.05
    system = np.vstack([weights.T, smooth * second_diff])   # (3+6, 8)
    target = np.concatenate([rgb, np.zeros(second_diff.shape[0])])
    color_curve, _residual = nnls(system, target)
    color_curve /= max(float(np.max(color_curve)), 1.0e-12)
    reflectivity = float(payload.get("reflectivity", 0.5))
    diffuse = float(payload.get("diffusion", 0.5))
    transmission = float(payload.get("transmission", 0.0))
    ior = float(payload.get("ior", 1.5))
    payload["bands"] = [
        {
            "center_hz": c_light / (lam * 1.0e-9),
            "bandwidth_hz": 4.0e13,
            "reflectance": float(np.clip(reflectivity * max(0.005, color_curve[i]), 0.0, 1.0)),
            "transmittance": float(np.clip(transmission, 0.0, 1.0)),
            "diffuse_frac": float(np.clip(diffuse, 0.0, 1.0)),
            "emission": 0.0,
            "reemission": 0.0,
            "ior_real": ior,
            "ior_imag": 0.0,
        }
        for i, lam in enumerate(wavelengths_nm)
    ]
    return payload


def compile_job(base_scene: Any, job: dict[str, Any]) -> tuple[Any, CompiledOrderReport]:
    """Compile one resolved job by replacing ``base_scene`` subject triangles."""
    _validate_resolved_job(job)
    from material_db import MaterialDatabase, MAX_SPECTRAL_BANDS

    base_tri = np.asarray(base_scene.verts, np.float64).reshape(-1, 3, 3)
    n_base = base_tri.shape[0]
    groups = base_scene.camera_tri_groups or {}
    remove = np.asarray(groups.get("object", np.zeros(0, np.int32)), np.int64).reshape(-1)
    remove = remove[(remove >= 0) & (remove < n_base)]
    # The lab's default subject ("orbiters") includes emissive demo spheres that
    # are tagged both as object geometry AND as flash sources.  A scene order
    # replaces the ENTIRE subject — including those in-frame demo emitters.
    # Photographic lighting (ring light, side-room flash) lives outside the
    # object group and is always retained.  Removed emitter triangles drop out
    # of the src_* arrays below via the remap validity mask.
    base_sources = np.asarray(base_scene.src_tri_idx, np.int64).reshape(-1)
    removed_subject_emitters = int(np.intersect1d(remove, base_sources).size)
    keep_mask = np.ones(n_base, dtype=bool)
    keep_mask[remove] = False
    keep_ids = np.flatnonzero(keep_mask)
    remap = np.full(n_base, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(keep_ids.size, dtype=np.int64)

    pose = camera_pose(job)
    canonical_sensor_ids = np.asarray(groups.get("sensor", np.zeros(0, np.int32)), np.int64)
    canonical_sensor_ids = canonical_sensor_ids[
        (canonical_sensor_ids >= 0) & (canonical_sensor_ids < n_base)
    ]
    if pose is not None and canonical_sensor_ids.size == 0:
        raise ValueError("authored camera pose requires a non-empty base sensor group")
    canonical_origin = (
        base_tri[canonical_sensor_ids].reshape(-1, 3).mean(axis=0)
        if canonical_sensor_ids.size
        else np.zeros(3, np.float64)
    )

    tri_parts = [np.asarray(base_tri[keep_ids], np.float64)]
    normal_parts = [np.asarray(base_scene.normals, np.float64)[keep_ids]]
    mat_parts = [np.asarray(base_scene.mat_idx, np.int32)[keep_ids]]
    object_ids: list[np.ndarray] = []

    authored_materials = job.get("materials", {})
    if not isinstance(authored_materials, dict):
        raise ValueError("materials must be an object")
    db = MaterialDatabase()
    material_names: list[str] = []
    material_local: dict[str, int] = {}

    def material_index(name: str, kind: str) -> int:
        if name not in material_local:
            authored = authored_materials.get(name, {})
            if not isinstance(authored, dict):
                raise ValueError(f"material {name!r} must be an object")
            material_local[name] = int(db.register(name, _material_payload(authored, kind)))
            material_names.append(name)
        return int(base_scene.mat_n_mats) + material_local[name]

    planes_by_id = {str(p["id"]): p for p in job["planes"]}
    plane_triangles = 0
    for plane in job["planes"]:
        tri = _world_to_canonical_camera(_box_plane_triangles(plane), pose, canonical_origin)
        start = sum(part.shape[0] for part in tri_parts)
        tri_parts.append(tri)
        normal_parts.append(_triangle_normals(tri))
        idx = material_index(str(plane.get("material", "matte_black")), "plane")
        mat_parts.append(np.full(tri.shape[0], idx, np.int32))
        object_ids.append(np.arange(start, start + tri.shape[0], dtype=np.int32))
        plane_triangles += int(tri.shape[0])

    target_plane = planes_by_id[str(job.get("geometry", {}).get("embed_plane", job["planes"][0]["id"]))]
    glyph_tri = _world_to_canonical_camera(
        _glyph_triangles(job, target_plane), pose, canonical_origin
    )
    glyph_start = sum(part.shape[0] for part in tri_parts)
    tri_parts.append(glyph_tri)
    normal_parts.append(_triangle_normals(glyph_tri))
    glyph_mat = material_index(str(job.get("geometry", {}).get("material", "glossy_red")), "glyph")
    mat_parts.append(np.full(glyph_tri.shape[0], glyph_mat, np.int32))
    object_ids.append(np.arange(glyph_start, glyph_start + glyph_tri.shape[0], dtype=np.int32))

    custom_buf = np.ascontiguousarray(db.build_mat_buf(), np.float32)
    if custom_buf.shape[0] % MAX_SPECTRAL_BANDS != 0:
        raise RuntimeError("authored material buffer has invalid spectral stride")
    mat_buf = np.ascontiguousarray(np.vstack([base_scene.mat_buf, custom_buf]), np.float32)

    flash_scale = float(job.get("flash", {}).get("intensity_scale", 1.0))
    if not math.isfinite(flash_scale) or flash_scale <= 0.0:
        raise ValueError("flash.intensity_scale must be positive")
    # Scale the actual spectral emission slots used by source triangles as well
    # as the host-side power bookkeeping.  Camera/lens geometry is untouched.
    if flash_scale != 1.0:
        source_old = np.asarray(base_scene.src_tri_idx, np.int64)
        source_old = source_old[(source_old >= 0) & (source_old < n_base)]
        # Scale only the RETAINED photographic emitters; removed subject-space
        # demo emitters must not have their (shared) materials touched.
        source_old = source_old[keep_mask[source_old]]
        source_mats = np.unique(np.asarray(base_scene.mat_idx, np.int32)[source_old])
        for mid in source_mats:
            lo = int(mid) * MAX_SPECTRAL_BANDS
            mat_buf[lo:lo + MAX_SPECTRAL_BANDS, 5] *= flash_scale

    tri_all = np.ascontiguousarray(np.concatenate(tri_parts, axis=0), np.float64)
    # Preserve the already-authored normals of retained camera/flash geometry;
    # the lab mesh intentionally contains a few zero-area bookkeeping faces.
    # Only newly compiled geometry is held to the non-degenerate contract.
    normals = np.ascontiguousarray(np.concatenate(normal_parts, axis=0), np.float64)
    mat_idx = np.ascontiguousarray(np.concatenate(mat_parts), np.int32)
    src_old = np.asarray(base_scene.src_tri_idx, np.int64).reshape(-1)
    src_valid = (src_old >= 0) & (src_old < n_base) & (remap[np.clip(src_old, 0, max(0, n_base-1))] >= 0)
    src_new = remap[src_old[src_valid]].astype(np.int32, copy=False)

    new_groups: dict[str, np.ndarray] = {}
    for name, ids in groups.items():
        if name == "object":
            continue
        old = np.asarray(ids, np.int64).reshape(-1)
        valid = (old >= 0) & (old < n_base)
        mapped = remap[old[valid]]
        new_groups[name] = np.ascontiguousarray(mapped[mapped >= 0], np.int32)
    new_groups["object"] = np.ascontiguousarray(np.concatenate(object_ids), np.int32)

    pts = tri_all.reshape(-1, 3)
    scene = base_scene.__class__(
        verts=np.ascontiguousarray(tri_all.reshape(-1, 9), np.float64),
        normals=normals,
        mat_idx=mat_idx,
        mat_buf=mat_buf,
        mat_n_mats=int(base_scene.mat_n_mats + custom_buf.shape[0] // MAX_SPECTRAL_BANDS),
        src_pos=np.ascontiguousarray(tri_all[src_new].mean(axis=1), np.float64),
        src_dir=np.ascontiguousarray(normals[src_new], np.float64),
        src_directivity=np.ascontiguousarray(np.asarray(base_scene.src_directivity)[src_valid], np.float64),
        src_area_m2=np.ascontiguousarray(np.asarray(base_scene.src_area_m2)[src_valid], np.float64),
        src_emit_W=np.ascontiguousarray(np.asarray(base_scene.src_emit_W)[src_valid] * flash_scale, np.float64),
        src_emit_rgb_W=np.ascontiguousarray(np.asarray(base_scene.src_emit_rgb_W)[src_valid] * flash_scale, np.float64),
        src_tri_idx=np.ascontiguousarray(src_new, np.int32),
        bounds_min=np.ascontiguousarray(pts.min(axis=0), np.float32),
        bounds_max=np.ascontiguousarray(pts.max(axis=0), np.float32),
        camera_tri_groups=new_groups,
    )
    report = CompiledOrderReport(
        job_id=str(job["id"]), token=str(job["token"]),
        plane_triangles=plane_triangles, glyph_triangles=int(glyph_tri.shape[0]),
        retained_base_triangles=int(keep_ids.size), total_triangles=int(tri_all.shape[0]),
        material_names=tuple(material_names), flash_scale=flash_scale,
        removed_subject_emitters=removed_subject_emitters,
        camera_position_m=(None if pose is None else tuple(float(v) for v in pose["position"])),
        camera_target_m=(None if pose is None else tuple(float(v) for v in pose["target"])),
    )
    return scene, report
