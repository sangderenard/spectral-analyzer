"""GPU-trained sensor work-value network and orthographic diagnostics.

The flat renderer is deliberately separate from the spectral renderer: it
produces supervised scene evidence, never camera radiance. Exact authored text
and scrambled variants may be used during training; runtime inference receives
only accumulated sensor RGB and exposure confidence.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
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


def render_flat_orthographic(
    job: dict[str, Any], width: int, height: int, *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Rasterize the formatted text plane as flat linear RGB on CUDA.

    The view covers the authored text box. Contour intersection is evaluated on
    the GPU with the even/odd rule; no lighting, lens, or camera approximation is
    mixed into this training reference.
    """

    import scene_orders

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("orthographic dimensions must be positive")
    dev = _device(device)
    geometry = job.get("geometry", {})
    contours = scene_orders._paragraph_contours(job) if "text_box_m" in geometry else (
        scene_orders._normalize_contours(
            scene_orders._font_contours(
                str(job["token"]), job.get("font", {}),
                int(geometry.get("outline_subdivisions", 4)),
            ),
            float(geometry.get("height_m", 0.2)),
        )
    )
    contours = scene_orders._sanitize_scaled_contours(
        contours, scene_orders.resolved_glyph_height(job)
    )
    box = np.asarray(
        geometry.get("text_box_m", [
            max(p[:, 0].max() for p in contours) - min(p[:, 0].min() for p in contours),
            max(p[:, 1].max() for p in contours) - min(p[:, 1].min() for p in contours),
        ]),
        np.float32,
    )
    yy = (0.5 - (torch.arange(height, device=dev, dtype=torch.float32) + 0.5) / height) * float(box[1])
    xx = ((torch.arange(width, device=dev, dtype=torch.float32) + 0.5) / width - 0.5) * float(box[0])
    py, px = torch.meshgrid(yy, xx, indexing="ij")
    inside = torch.zeros((height, width), dtype=torch.bool, device=dev)
    # Even/odd fill is parity across every contour segment, including holes.
    # Flattening first avoids one CUDA launch sequence per glyph contour.
    starts = np.concatenate([contour[:-1] for contour in contours], axis=0)
    ends = np.concatenate([contour[1:] for contour in contours], axis=0)
    p0 = torch.as_tensor(starts, dtype=torch.float32, device=dev)
    p1 = torch.as_tensor(ends, dtype=torch.float32, device=dev)
    for start in range(0, int(p0.shape[0]), 128):
        sl = slice(start, start + 128)
        ax0, ay0 = p0[sl, 0, None, None], p0[sl, 1, None, None]
        ax1, ay1 = p1[sl, 0, None, None], p1[sl, 1, None, None]
        intersects_y = (ay0 > py) != (ay1 > py)
        x_cross = ax0 + (py - ay0) * (ax1 - ax0) / (ay1 - ay0 + 1.0e-20)
        inside ^= torch.sum(intersects_y & (px < x_cross), dim=0).remainder(2).bool()

    plane_name = str(job.get("planes", [{}])[0].get("material", "quiet_background"))
    text_name = str(geometry.get("material", "text_surface"))
    background = torch.as_tensor(
        _material_rgb(job, plane_name, [0.018, 0.021, 0.022]), device=dev
    )
    foreground = torch.as_tensor(
        _material_rgb(job, text_name, [0.82, 0.76, 0.62]), device=dev
    )
    return torch.where(inside[..., None], foreground, background).contiguous()


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


def _scrambled(text: str, rng: random.Random) -> str:
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    chars = [text[i] for i in positions]
    rng.shuffle(chars)
    result = list(text)
    for position, char in zip(positions, chars):
        result[position] = char
    return "".join(result)


def _simulate_evidence(reference: torch.Tensor, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = reference.shape
    coarse = torch.rand((batch, 1, max(2, height // 12), max(2, width // 12)), device=reference.device)
    density = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    density = torch.clamp((density - 0.15) * 1.25, 0.0, 1.0)
    maximum = torch.randint(2, 33, (batch, 1, 1, 1), device=reference.device)
    exposure = torch.floor(density * maximum).float()
    sigma = 0.24 / torch.sqrt(exposure + 1.0)
    observed = torch.clamp(reference + torch.randn_like(reference) * sigma, 0.0, 1.0)
    observed = torch.where(exposure > 0.0, observed, torch.zeros_like(observed))
    return observed, exposure


def _work_target(observed: torch.Tensor, exposure: torch.Tensor,
                 reference: torch.Tensor) -> torch.Tensor:
    error = torch.mean((observed - reference).square(), dim=1, keepdim=True)
    gray = reference.mean(dim=1, keepdim=True)
    gx = F.pad(torch.abs(gray[..., :, 1:] - gray[..., :, :-1]), (0, 1, 0, 0))
    gy = F.pad(torch.abs(gray[..., 1:, :] - gray[..., :-1, :]), (0, 0, 0, 1))
    edge = torch.clamp(gx + gy, 0.0, 1.0)
    remaining = torch.rsqrt(exposure + 1.0)
    holes = (exposure <= 0.0).float()
    target = F.avg_pool2d(error, 5, stride=1, padding=2) * remaining
    target = target + 0.2 * edge * remaining + 0.35 * holes
    peak = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    return target / peak


@dataclass(frozen=True)
class PriorityTrainingArtifacts:
    model_path: str
    flat_reference_path: str
    priority_overlay_path: str
    metadata_path: str
    final_loss: float


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, np.uint8), "RGB").save(path)


def _heatmap(values: np.ndarray, base: np.ndarray) -> np.ndarray:
    v = np.clip(values / max(float(values.max()), 1.0e-8), 0.0, 1.0)
    heat = np.stack([v, np.sqrt(v) * 0.35, 1.0 - v], axis=-1)
    return np.clip(0.30 * base + 0.70 * heat, 0.0, 1.0)


def train_scene_priority_network(
    job: dict[str, Any], output_dir: str, *, width: int, height: int,
    steps: int = 240, seed: int = 17,
) -> PriorityTrainingArtifacts:
    """Train on exact/scrambled flat scenes and save an inspectable overlay."""

    if steps <= 0:
        raise ValueError("training steps must be positive")
    os.makedirs(output_dir, exist_ok=True)
    dev = _device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    variants = [job]
    for _ in range(3):
        variant = dict(job)
        variant["token"] = _scrambled(str(job["token"]), rng)
        variants.append(variant)
    references = torch.stack([
        render_flat_orthographic(item, width, height, device=dev).permute(2, 0, 1)
        for item in variants
    ])
    model = SensorWorkValueNet().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    final_loss = 0.0
    for _ in range(int(steps)):
        indices = torch.randint(0, references.shape[0], (8,), device=dev)
        truth = references[indices]
        observed, exposure = _simulate_evidence(truth, truth.shape[0])
        target = _work_target(observed, exposure, truth)
        predicted = model(sensor_features(observed, exposure))
        loss = F.smooth_l1_loss(predicted, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    exact = references[:1]
    observed, exposure = _simulate_evidence(exact, 1)
    with torch.no_grad():
        priority = model(sensor_features(observed, exposure))[0, 0]
    flat = exact[0].permute(1, 2, 0).detach().cpu().numpy()
    priority_np = priority.detach().cpu().numpy()
    params = model.export_glsl_parameters()
    model_path = os.path.abspath(os.path.join(output_dir, "sensor_priority_network.npz"))
    np.savez(model_path, parameters=params, hidden_channels=np.int32(HIDDEN_CHANNELS),
             input_channels=np.int32(INPUT_CHANNELS), kernel_size=np.int32(KERNEL_SIZE))
    flat_path = os.path.abspath(os.path.join(output_dir, "orthographic_flat.png"))
    overlay_path = os.path.abspath(os.path.join(output_dir, "priority_overlay.png"))
    metadata_path = os.path.abspath(os.path.join(output_dir, "priority_training.json"))
    _save_rgb(flat_path, flat)
    _save_rgb(overlay_path, _heatmap(priority_np, flat))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": 1,
            "architecture": "conv3x3-4x8-relu-conv1x1-softplus",
            "device": str(dev),
            "training_steps": int(steps),
            "seed": int(seed),
            "exact_text": str(job["token"]),
            "scrambled_variants": [str(item["token"]) for item in variants[1:]],
            "parameter_count": int(params.size),
            "final_loss": final_loss,
        }, handle, indent=2)
    return PriorityTrainingArtifacts(
        model_path=model_path,
        flat_reference_path=flat_path,
        priority_overlay_path=overlay_path,
        metadata_path=metadata_path,
        final_loss=final_loss,
    )
