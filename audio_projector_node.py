"""Audio projector node for graph-solver outputs.

Mixers stay complex analytic accumulators.  AudioProjectorNode is the graph
boundary that turns a complex analytic stream into an audio-domain projection
and can write that projection to a WAV file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import wave

import torch
import torch.nn as nn
from torch import Tensor

from graph_solver import (
    NodeArchetype,
    SemanticPortContract,
    TensorNode,
    _CDTYPE,
)


def project_complex_audio(x: Tensor, projection: str = "real") -> Tensor:
    """Project a complex analytic signal to a real audio-domain tensor."""
    mode = str(projection or "real").lower()
    z = x.to(_CDTYPE)
    if mode == "real":
        return z.real.to(torch.float64)
    if mode == "imag":
        return z.imag.to(torch.float64)
    if mode in ("abs", "magnitude"):
        return z.abs().to(torch.float64)
    raise ValueError(f"Unknown audio projection mode: {projection!r}")


def _time_series_1d(x: Tensor, *, part: str = "real") -> Tensor:
    """Extract the first batch/channel time series from a solver payload."""
    z = x.to(_CDTYPE).detach().cpu()
    if z.dim() >= 3:
        series = z[0, :, 0]
    elif z.dim() == 2:
        series = z[:, 0]
    else:
        series = z.reshape(-1)
    if part == "real":
        return series.real.to(torch.float64)
    if part == "imag":
        return series.imag.to(torch.float64)
    return series.abs().to(torch.float64)


def _normalise_pcm(audio: Tensor, *, target_peak: float) -> Tensor:
    y = audio.detach().cpu().to(torch.float64)
    peak = float(y.abs().max().item()) if y.numel() else 0.0
    if peak > 1e-12:
        y = y * (float(target_peak) / peak)
    return y.clamp(-1.0, 1.0)


def write_wav_pcm16(
    path: str | Path,
    audio: Tensor,
    *,
    sample_rate: float,
    normalize: bool = True,
    target_peak: float = 0.95,
) -> Path:
    """Write mono or stereo float audio tensor to PCM16 WAV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = audio.detach().cpu().to(torch.float64)
    if y.dim() == 1:
        channels = 1
        frames = y.reshape(-1, 1)
    elif y.dim() == 2:
        channels = int(y.shape[1])
        frames = y
    else:
        raise ValueError("audio must be a 1-D mono or 2-D [frames, channels] tensor")
    if normalize:
        frames = _normalise_pcm(frames, target_peak=target_peak)
    else:
        frames = frames.clamp(-1.0, 1.0)
    pcm = torch.round(frames * 32767.0).to(torch.int16).contiguous().numpy()
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(round(float(sample_rate))))
        wf.writeframes(pcm.tobytes())
    return out_path


class AudioProjectorNode(nn.Module):
    """Project complex graph audio into real PCM-facing audio.

    Parameters
    ----------
    key:
        Tensor node key for the projected output.
    sample_rate:
        WAV sample rate when ``wav_path`` is set.
    projection:
        Graph output projection: ``"real"`` (default), ``"imag"``, or
        ``"magnitude"``.  The node returns this projection as complex128 with
        zero imaginary part so downstream graph code remains dtype-uniform.
    wav_path:
        Optional WAV destination.  Schedule-mode payloads are written when the
        transform runs.
    wav_channels:
        ``"mono"`` writes the selected projection. ``"stereo_quadrature"``
        writes real/imag as left/right for inspecting the analytic signal.
    """

    def __init__(
        self,
        key: str = "audio_out",
        *,
        sample_rate: float = 48_000.0,
        projection: str = "real",
        wav_path: str | Path | None = None,
        wav_channels: str = "mono",
        normalize: bool = True,
        target_peak: float = 0.95,
        layer: str = "audio",
    ) -> None:
        super().__init__()
        self.key = str(key)
        self.sample_rate = float(sample_rate)
        self.projection = str(projection)
        self.wav_path = Path(wav_path) if wav_path is not None else None
        self.wav_channels = str(wav_channels)
        self.normalize = bool(normalize)
        self.target_peak = float(target_peak)
        self.layer = str(layer)
        self.last_write_path: Optional[Path] = None

    def _wav_tensor(self, x: Tensor) -> Tensor:
        channels = self.wav_channels.lower()
        if channels == "mono":
            return _time_series_1d(x, part=self.projection.lower())
        if channels in ("stereo_quadrature", "real_imag"):
            left = _time_series_1d(x, part="real")
            right = _time_series_1d(x, part="imag")
            return torch.stack((left, right), dim=1)
        raise ValueError(f"Unknown wav channel mode: {self.wav_channels!r}")

    def _transform(self, x: Tensor) -> Tensor:
        projected = project_complex_audio(x, self.projection)
        if self.wav_path is not None and x.dim() >= 2:
            self.last_write_path = write_wav_pcm16(
                self.wav_path,
                self._wav_tensor(x),
                sample_rate=self.sample_rate,
                normalize=self.normalize,
                target_peak=self.target_peak,
            )
        return projected.to(_CDTYPE)

    def build_node(self) -> TensorNode:
        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=self._transform,
            analytic_module=self,
            archetype=NodeArchetype(
                semantic_ports=(
                    SemanticPortContract(
                        key="audio_in",
                        label="Audio In",
                        direction="in",
                        domain="signal",
                        semantic_role="audio_projection_source",
                    ),
                    SemanticPortContract(
                        key="audio_out",
                        label="Audio Out",
                        direction="out",
                        domain="audio",
                        semantic_role="projected_audio",
                    ),
                ),
            ),
        )

