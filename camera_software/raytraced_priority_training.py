"""Self-supervised work-value training from real progressive ray-traced states."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .sensor_priority_network import (
    HIDDEN_CHANNELS,
    INPUT_CHANNELS,
    KERNEL_SIZE,
    SensorWorkValueNet,
    _device,
    sensor_features,
)


_PASS_RE = re.compile(r"^sum_(\d+)_z\d+_s\d+(?:_n\d+)?\.npy$")


@dataclass(frozen=True)
class RaytracedSensorState:
    pass_index: int
    sensor_sum: np.ndarray
    exposure_weight: np.ndarray


@dataclass(frozen=True)
class RaytracedWorkExample:
    pass_index: int
    camera_rgb: np.ndarray
    exposure_weight: np.ndarray
    measured_improvement: np.ndarray


@dataclass(frozen=True)
class RaytracedTrainingResult:
    model_path: str
    metadata_path: str
    example_count: int
    final_loss: float


def normalized_camera_rgb(sensor_sum: np.ndarray, exposure_weight: np.ndarray) -> np.ndarray:
    """Match the normalized RGB received by runtime GLSL inference."""
    sums = np.asarray(sensor_sum, np.float32)
    weights = np.asarray(exposure_weight, np.float32)
    if sums.ndim != 3 or sums.shape[2] != 3 or weights.shape != sums.shape[:2]:
        raise ValueError("expected sensor sum HxWx3 and matching exposure weights")
    result = np.zeros_like(sums)
    np.divide(sums, weights[..., None], out=result, where=weights[..., None] > 0.0)
    return np.maximum(result, 0.0)


def load_raytraced_states(progress_dir: str) -> list[RaytracedSensorState]:
    """Load exact sum/weight pairs emitted by progressive GPU exposure."""
    root = Path(progress_dir)
    states: list[RaytracedSensorState] = []
    for sum_path in root.glob("sum_*.npy"):
        match = _PASS_RE.match(sum_path.name)
        if match is None:
            continue
        suffix = sum_path.name[len("sum_"):]
        weight_path = root / f"weight_{suffix}"
        if not weight_path.is_file():
            continue
        states.append(RaytracedSensorState(
            pass_index=int(match.group(1)),
            sensor_sum=np.ascontiguousarray(np.load(sum_path, allow_pickle=False), np.float32),
            exposure_weight=np.ascontiguousarray(
                np.load(weight_path, allow_pickle=False), np.float32
            ),
        ))
    states.sort(key=lambda state: state.pass_index)
    if len(states) < 2:
        raise ValueError(f"{root} contains fewer than two sum/weight exposure states")
    shape = states[0].sensor_sum.shape
    if any(state.sensor_sum.shape != shape for state in states):
        raise ValueError("ray-traced states do not share one sensor shape")
    return states


def measured_work_examples(
    states: Sequence[RaytracedSensorState], *, smoothing: int = 5,
) -> list[RaytracedWorkExample]:
    """Label work by the actual next-layer error reduction toward later evidence.

    The last available exposure is a noisy teacher, not synthetic ground truth.
    Broad/random coverage is therefore important during collection: it supplies
    counterexamples in regions the current learned scheduler would not choose.
    """
    if len(states) < 2:
        raise ValueError("at least two progressive states are required")
    if smoothing <= 0 or smoothing % 2 == 0:
        raise ValueError("smoothing must be a positive odd integer")
    images = [normalized_camera_rgb(s.sensor_sum, s.exposure_weight) for s in states]
    teacher = images[-1]
    examples: list[RaytracedWorkExample] = []
    for index in range(len(states) - 1):
        before, after = images[index], images[index + 1]
        before_error = np.mean(np.square(before - teacher), axis=2)
        after_error = np.mean(np.square(after - teacher), axis=2)
        improvement = np.maximum(before_error - after_error, 0.0)
        target = torch.as_tensor(improvement)[None, None]
        target = F.avg_pool2d(
            target, smoothing, stride=1, padding=smoothing // 2
        )[0, 0].numpy()
        peak = max(float(np.max(target)), 1.0e-12)
        target = np.ascontiguousarray(target / peak, np.float32)
        examples.append(RaytracedWorkExample(
            pass_index=states[index].pass_index,
            camera_rgb=before,
            exposure_weight=states[index].exposure_weight,
            measured_improvement=target,
        ))
    return examples


def train_from_raytraced_examples(
    examples: Sequence[RaytracedWorkExample], output_dir: str, *,
    steps: int = 400, seed: int = 17, resume_model: str = "",
) -> RaytracedTrainingResult:
    """Train the runtime 2-D convolution on measured ray-traced improvement."""
    if not examples:
        raise ValueError("ray-traced training requires at least one example")
    if steps <= 0:
        raise ValueError("training steps must be positive")
    os.makedirs(output_dir, exist_ok=True)
    dev = _device()
    torch.manual_seed(int(seed))
    model = SensorWorkValueNet().to(dev)
    if str(resume_model).strip():
        archive = np.load(os.path.abspath(resume_model), allow_pickle=False)
        model.load_glsl_parameters(archive["parameters"])
    rgb = torch.stack([
        torch.as_tensor(item.camera_rgb, device=dev).permute(2, 0, 1)
        for item in examples
    ])
    exposure = torch.stack([
        torch.as_tensor(item.exposure_weight, device=dev)[None]
        for item in examples
    ])
    targets = torch.stack([
        torch.as_tensor(item.measured_improvement, device=dev)[None]
        for item in examples
    ])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    final_loss = 0.0
    batch_size = min(8, len(examples))
    for _ in range(int(steps)):
        indices = torch.randint(0, len(examples), (batch_size,), device=dev)
        prediction = model(sensor_features(rgb[indices], exposure[indices]))
        loss = F.smooth_l1_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    model_path = os.path.abspath(os.path.join(output_dir, "raytraced_priority_network.npz"))
    metadata_path = os.path.abspath(os.path.join(output_dir, "raytraced_priority_training.json"))
    parameters = model.export_glsl_parameters()
    np.savez(
        model_path,
        parameters=parameters,
        hidden_channels=np.int32(HIDDEN_CHANNELS),
        input_channels=np.int32(INPUT_CHANNELS),
        kernel_size=np.int32(KERNEL_SIZE),
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": 1,
            "training_source": "progressive-spectral-raytraced-sensor-sum-and-weight",
            "target": "positive-next-layer-error-reduction-toward-last-exposure",
            "camera_image_is_input": True,
            "architecture": "conv3x3-4x8-relu-conv1x1-softplus",
            "example_count": len(examples),
            "training_steps": int(steps),
            "seed": int(seed),
            "resumed_from": os.path.abspath(resume_model) if resume_model else "",
            "final_loss": final_loss,
        }, handle, indent=2)
    return RaytracedTrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        example_count=len(examples),
        final_loss=final_loss,
    )
