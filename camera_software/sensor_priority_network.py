"""Runtime sensor work-value network and presentation-only orthographic preview."""
from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HIDDEN_CHANNELS = 8
INPUT_CHANNELS = 4
KERNEL_SIZE = 3
PARAMETER_COUNT = HIDDEN_CHANNELS * (INPUT_CHANNELS * 9 + 1) + HIDDEN_CHANNELS + 1


def _device(requested: str | torch.device | None = None) -> torch.device:
    result = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if result.type != "cuda":
        raise RuntimeError("sensor priority training/inference requires a CUDA GPU")
    return result


def _material_rgb(job: dict[str, Any], name: str, fallback: Sequence[float]) -> np.ndarray:
    authored = job.get("materials", {}).get(name, {})
    return np.clip(np.asarray(authored.get("albedo_rgb", fallback), np.float32), 0.0, 1.0)


def _orthographic_scene_geometry(
    job: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact authored triangles and camera-aligned projection frame."""

    import scene_orders

    pose = scene_orders.camera_pose(job)
    target_id = str(job.get("geometry", {}).get(
        "embed_plane", job["planes"][0]["id"]
    ))
    target_plane = next(
        plane for plane in job["planes"] if str(plane["id"]) == target_id
    )
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    target_triangles = None
    for plane in job["planes"]:
        plane_triangles = scene_orders._box_plane_triangles(plane)
        triangles.append(plane_triangles)
        color = _material_rgb(
            job, str(plane.get("material", "quiet_background")),
            [0.018, 0.021, 0.022],
        )
        colors.append(np.repeat(color[None, :], plane_triangles.shape[0], axis=0))
        if str(plane["id"]) == target_id:
            target_triangles = plane_triangles
    authored_objects = job.get("objects", ())
    if authored_objects:
        planes_by_id = {str(plane["id"]): plane for plane in job["planes"]}
        for raw_object in authored_objects:
            if not bool(raw_object.get("enabled", True)):
                continue
            object_job = copy.deepcopy(job)
            object_job.pop("objects", None)
            object_job["token"] = str(raw_object["token"])
            object_job["font"] = scene_orders._deep_merge(
                dict(job.get("font", {})), dict(raw_object.get("font", {}))
            )
            object_job["geometry"] = scene_orders._deep_merge(
                dict(job.get("geometry", {})), dict(raw_object.get("geometry", {}))
            )
            object_job["geometry"]["embed_plane"] = str(
                raw_object.get(
                    "embed_plane",
                    object_job["geometry"].get("embed_plane", target_id),
                )
            )
            glyph_triangles = scene_orders._glyph_triangles(
                object_job,
                planes_by_id[object_job["geometry"]["embed_plane"]],
            )
            triangles.append(glyph_triangles)
            glyph_color = _material_rgb(
                job,
                str(object_job["geometry"].get("material", "text_surface")),
                [0.82, 0.76, 0.62],
            )
            colors.append(np.repeat(
                glyph_color[None, :], glyph_triangles.shape[0], axis=0
            ))
    else:
        glyph_triangles = scene_orders._glyph_triangles(job, target_plane)
        triangles.append(glyph_triangles)
        glyph_color = _material_rgb(
            job, str(job.get("geometry", {}).get("material", "text_surface")),
            [0.82, 0.76, 0.62],
        )
        colors.append(np.repeat(
            glyph_color[None, :], glyph_triangles.shape[0], axis=0
        ))
    if target_triangles is None:
        raise ValueError(f"embed plane {target_id!r} has no geometry")

    if pose is None:
        center, normal, right, up = scene_orders._plane_basis(target_plane)
        camera_position = center + normal
        forward = -normal
    else:
        camera_position = pose["position"]
        forward, right, up = pose["fwd"], pose["right"], pose["up"]
    return (
        np.ascontiguousarray(np.concatenate(triangles, axis=0), np.float32),
        np.ascontiguousarray(np.concatenate(colors, axis=0), np.float32),
        np.ascontiguousarray(camera_position, np.float32),
        np.ascontiguousarray(forward, np.float32),
        np.ascontiguousarray(np.stack([right, up]), np.float32),
    )


def orthographic_scene_bounds(
    job: dict[str, Any], width: int, height: int,
) -> tuple[float, float, float, float]:
    """Camera-aligned flat view bounds for the requested sensor-region fraction."""

    import scene_orders

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("orthographic dimensions must be positive")
    plane_triangles = np.concatenate([
        scene_orders._box_plane_triangles(plane) for plane in job["planes"]
    ], axis=0)
    pose = scene_orders.camera_pose(job)
    if pose is None:
        _center, _normal, right, up = scene_orders._plane_basis(job["planes"][0])
    else:
        right, up = pose["right"], pose["up"]
    projected_x = plane_triangles.reshape(-1, 3) @ right
    projected_y = plane_triangles.reshape(-1, 3) @ up
    full_left, full_right = float(projected_x.min()), float(projected_x.max())
    full_bottom, full_top = float(projected_y.min()), float(projected_y.max())

    runtime = scene_orders.order_runtime_settings(job)
    region = runtime["region"]
    u0 = float(region["x"]) / float(runtime["full_width"])
    u1 = float(region["x"] + region["width"]) / float(runtime["full_width"])
    v0 = float(region["y"]) / float(runtime["full_height"])
    v1 = float(region["y"] + region["height"]) / float(runtime["full_height"])
    left = full_left + u0 * (full_right - full_left)
    right_bound = full_left + u1 * (full_right - full_left)
    top = full_top - v0 * (full_top - full_bottom)
    bottom = full_top - v1 * (full_top - full_bottom)

    # A standard orthographic camera has square world units. Expand one axis
    # to the output aspect instead of stretching the scene geometry.
    center_x, center_y = 0.5 * (left + right_bound), 0.5 * (bottom + top)
    view_width, view_height = right_bound - left, top - bottom
    output_aspect = float(width) / float(height)
    if view_width / view_height < output_aspect:
        view_width = view_height * output_aspect
    else:
        view_height = view_width / output_aspect
    return (
        center_x - 0.5 * view_width,
        center_x + 0.5 * view_width,
        center_y - 0.5 * view_height,
        center_y + 0.5 * view_height,
    )


def render_flat_orthographic(
    job: dict[str, Any], width: int, height: int, *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """GPU-rasterize the exact scene triangles with flat material colors.

    This is a conventional camera-aligned orthographic triangle projection. It
    uses the requested sensor-region fraction for framing and does not invoke
    ray tracing, lighting, lens simulation, font contours, or perspective.
    """

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("orthographic dimensions must be positive")
    dev = _device(device)
    triangles_np, colors_np, camera_position_np, forward_np, axes_np = (
        _orthographic_scene_geometry(job)
    )
    left, right_bound, bottom, top = orthographic_scene_bounds(job, width, height)
    yy = top - (torch.arange(height, device=dev, dtype=torch.float32) + 0.5) * (
        (top - bottom) / height
    )
    xx = left + (torch.arange(width, device=dev, dtype=torch.float32) + 0.5) * (
        (right_bound - left) / width
    )
    py, px = torch.meshgrid(yy, xx, indexing="ij")
    triangles = torch.as_tensor(triangles_np, device=dev)
    colors = torch.as_tensor(colors_np, device=dev)
    camera_position = torch.as_tensor(camera_position_np, device=dev)
    forward = torch.as_tensor(forward_np, device=dev)
    axes = torch.as_tensor(axes_np, device=dev)
    relative = triangles - camera_position[None, None, :]
    projected = torch.stack([
        torch.sum(triangles * axes[0][None, None, :], dim=-1),
        torch.sum(triangles * axes[1][None, None, :], dim=-1),
    ], dim=-1)
    depths = torch.sum(relative * forward[None, None, :], dim=-1)
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=dev)
    triangle_min = projected.amin(dim=1)
    triangle_max = projected.amax(dim=1)
    # Hardware-style coarse binning: each exact triangle is tested only against
    # the small raster tiles touched by its projected bounding box.
    tile_size = 32
    for pixel_y0 in range(0, height, tile_size):
        pixel_y1 = min(pixel_y0 + tile_size, height)
        for pixel_x0 in range(0, width, tile_size):
            pixel_x1 = min(pixel_x0 + tile_size, width)
            tile_px = px[pixel_y0:pixel_y1, pixel_x0:pixel_x1]
            tile_py = py[pixel_y0:pixel_y1, pixel_x0:pixel_x1]
            tile_left = float(xx[pixel_x0]) - 0.5 * (right_bound - left) / width
            tile_right = float(xx[pixel_x1 - 1]) + 0.5 * (right_bound - left) / width
            tile_top = float(yy[pixel_y0]) + 0.5 * (top - bottom) / height
            tile_bottom = float(yy[pixel_y1 - 1]) - 0.5 * (top - bottom) / height
            overlaps = (
                (triangle_max[:, 0] >= tile_left)
                & (triangle_min[:, 0] <= tile_right)
                & (triangle_max[:, 1] >= tile_bottom)
                & (triangle_min[:, 1] <= tile_top)
            )
            triangle_ids = torch.nonzero(overlaps, as_tuple=False).flatten()
            if triangle_ids.numel() == 0:
                continue
            tile_depth = torch.full(tile_px.shape, torch.inf, device=dev)
            tile_image = torch.zeros(
                (*tile_px.shape, 3), dtype=torch.float32, device=dev
            )
            for batch_start in range(0, int(triangle_ids.numel()), 64):
                ids = triangle_ids[batch_start:batch_start + 64]
                p = projected[ids]
                x0, y0 = p[:, 0, 0, None, None], p[:, 0, 1, None, None]
                x1, y1 = p[:, 1, 0, None, None], p[:, 1, 1, None, None]
                x2, y2 = p[:, 2, 0, None, None], p[:, 2, 1, None, None]
                denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                valid = torch.abs(denominator) > 1.0e-14
                inv = torch.where(valid, 1.0 / denominator, torch.zeros_like(denominator))
                w0 = ((y1 - y2) * (tile_px - x2) + (x2 - x1) * (tile_py - y2)) * inv
                w1 = ((y2 - y0) * (tile_px - x2) + (x0 - x2) * (tile_py - y2)) * inv
                w2 = 1.0 - w0 - w1
                covered = (
                    valid & (w0 >= -1.0e-6) & (w1 >= -1.0e-6) & (w2 >= -1.0e-6)
                )
                d = depths[ids]
                candidate_depth = (
                    w0 * d[:, 0, None, None]
                    + w1 * d[:, 1, None, None]
                    + w2 * d[:, 2, None, None]
                )
                candidate_depth = torch.where(
                    covered & (candidate_depth > 0.0), candidate_depth, torch.inf
                )
                nearest_depth, nearest_local = torch.min(candidate_depth, dim=0)
                replace_pixels = nearest_depth < tile_depth
                nearest_colors = colors[ids][nearest_local]
                tile_image = torch.where(
                    replace_pixels[..., None], nearest_colors, tile_image
                )
                tile_depth = torch.minimum(tile_depth, nearest_depth)
            image[pixel_y0:pixel_y1, pixel_x0:pixel_x1] = tile_image
    return image.contiguous()


class SensorWorkValueNet(nn.Module):
    """One 3x3 convolution plus a 1x1 positive work-value head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, 3, padding=1)
        self.head = nn.Conv2d(HIDDEN_CHANNELS, 1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.head(F.relu(self.features(features))))

    def export_glsl_parameters(self) -> np.ndarray:
        parts = (
            self.features.weight.detach().reshape(-1),
            self.features.bias.detach().reshape(-1),
            self.head.weight.detach().reshape(-1),
            self.head.bias.detach().reshape(-1),
        )
        result = torch.cat(parts).float().cpu().numpy()
        if result.size != PARAMETER_COUNT:
            raise RuntimeError(f"unexpected priority network size {result.size}")
        return np.ascontiguousarray(result, dtype=np.float32)

    def load_glsl_parameters(self, parameters: np.ndarray) -> None:
        """Load the fixed exported ABI back into the PyTorch model."""
        values = np.asarray(parameters, np.float32).reshape(-1)
        if values.size != PARAMETER_COUNT or not np.all(np.isfinite(values)):
            raise ValueError(f"expected {PARAMETER_COUNT} finite network parameters")
        offset = 0
        with torch.no_grad():
            for tensor in (
                self.features.weight, self.features.bias,
                self.head.weight, self.head.bias,
            ):
                count = tensor.numel()
                tensor.copy_(torch.as_tensor(
                    values[offset:offset + count], device=tensor.device
                ).reshape_as(tensor))
                offset += count


def sensor_features(rgb: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
    """Match the four normalized channels consumed by GLSL inference."""

    if rgb.ndim != 4 or rgb.shape[1] != 3 or exposure.shape != rgb[:, :1].shape:
        raise ValueError("expected RGB (N,3,H,W) and exposure (N,1,H,W)")
    positive = torch.clamp(rgb, min=0.0)
    total = positive.sum(dim=1, keepdim=True)
    luma = (positive * positive.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    chroma_r = positive[:, 0:1] / (total + 1.0e-6)
    chroma_g = positive[:, 1:2] / (total + 1.0e-6)
    compressed_luma = luma / (luma + 0.05)
    confidence = torch.clamp(torch.log1p(torch.clamp(exposure, min=0.0)) / math.log(65.0), 0.0, 1.0)
    return torch.cat([compressed_luma, chroma_r, chroma_g, confidence], dim=1)
