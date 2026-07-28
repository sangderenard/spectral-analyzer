"""Small fftfree-powered control stream for audio-reactive visual systems."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpectralControls:
    loudness: float
    bass: float
    low_mid: float
    high_mid: float
    treble: float


def _default_fftfree_dll() -> Path:
    configured = os.environ.get("FFTFREE_DLL")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parent.parent
        / "fftfree"
        / "build"
        / "Release"
        / "fft_cffi.dll"
    )


class FftfreeSpectrum:
    """Reusable real-input fftfree plan exposed as NumPy magnitude vectors."""

    def __init__(self, size: int = 2048, *, threads: int = 1) -> None:
        if size < 8 or size & (size - 1):
            raise ValueError("fftfree analysis size must be a power of two")
        dll_path = _default_fftfree_dll()
        if not dll_path.is_file():
            raise FileNotFoundError(
                f"fftfree runtime not found at {dll_path}; set FFTFREE_DLL"
            )
        self.size = int(size)
        self._dll = ctypes.CDLL(str(dll_path))
        self._dll.fft_init_full.argtypes = (
            ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        )
        self._dll.fft_init_full.restype = ctypes.c_void_p
        self._dll.fft_execute.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        )
        self._dll.fft_execute.restype = ctypes.c_int
        self._dll.fft_free.argtypes = (ctypes.c_void_p,)
        self._dll.fft_free.restype = None
        self._handle = self._dll.fft_init_full(
            self.size,
            max(1, int(threads)),
            0, 0, 0, 0, None, 0,
            2, self.size, self.size, 0,
            1, 1, 0, 1,
            0, 0, 1, 0, 1,
        )
        if not self._handle:
            raise RuntimeError("fftfree could not initialize its R2C plan")
        self._real = np.empty(self.size, dtype=np.float32)
        self._imag = np.empty(self.size, dtype=np.float32)
        self._magnitude = np.empty(self.size, dtype=np.float32)

    def magnitude(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float32)
        if values.shape != (self.size,):
            raise ValueError(f"expected {self.size} samples, got {values.shape}")
        values = np.ascontiguousarray(values)
        ok = self._dll.fft_execute(
            self._handle,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._imag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._magnitude.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.size,
        )
        if not ok:
            raise RuntimeError("fftfree analysis execution failed")
        return self._magnitude[: self.size // 2 + 1]

    def close(self) -> None:
        if self._handle:
            self._dll.fft_free(self._handle)
            self._handle = None


class AudioReactiveControlStream:
    """Looping audio file decoded by Pluck and analyzed through fftfree."""

    def __init__(
        self,
        path: str | Path,
        *,
        fft_size: int = 2048,
        gain: float = 1.0,
        smoothing: float = 0.22,
    ) -> None:
        import soundfile as sf

        self.path = Path(path)
        audio, sample_rate = sf.read(
            self.path, dtype="float32", always_2d=True
        )
        if not len(audio):
            raise ValueError(f"audio file is empty: {self.path}")
        self.samples = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
        self.sample_rate = int(sample_rate)
        self.duration = len(self.samples) / self.sample_rate
        self.gain = float(gain)
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.fft = FftfreeSpectrum(fft_size)
        self.window = np.asarray(np.hanning(fft_size), dtype=np.float32)
        self.frequencies = (
            np.arange(fft_size // 2 + 1, dtype=np.float32)
            * self.sample_rate
            / fft_size
        )
        self._envelope = np.zeros(5, dtype=np.float64)
        self._reference = np.full(5, 1e-6, dtype=np.float64)

    def _window_at(self, time_sec: float) -> np.ndarray:
        center = int((float(time_sec) % self.duration) * self.sample_rate)
        offsets = np.arange(self.fft.size) + center - self.fft.size // 2
        return self.samples[np.mod(offsets, len(self.samples))] * self.window

    def sample(self, time_sec: float) -> SpectralControls:
        window = self._window_at(time_sec)
        magnitude = self.fft.magnitude(window)
        bands = ((20, 120), (120, 500), (500, 2500), (2500, 12000))
        energy = [float(np.sqrt(np.mean(window * window)))]
        for lower, upper in bands:
            mask = (self.frequencies >= lower) & (self.frequencies < upper)
            values = magnitude[mask]
            energy.append(
                float(np.sqrt(np.mean(values * values))) if len(values) else 0.0
            )
        energy = np.asarray(energy, dtype=np.float64) * self.gain
        alpha = self.smoothing
        self._envelope += alpha * (energy - self._envelope)
        self._reference = np.maximum(
            self._reference * 0.997,
            self._envelope,
        )
        controls = np.clip(
            self._envelope / np.maximum(self._reference, 1e-9), 0.0, 1.0
        )
        return SpectralControls(*map(float, controls))

    def close(self) -> None:
        self.fft.close()
