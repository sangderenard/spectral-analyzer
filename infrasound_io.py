"""Infrasound / hydrophone / seismic data ingestion.

Thin wrappers that load unmodified professional data into
(torch.Tensor, sample_rate, metadata) tuples ready for ``cwt()``
or ``audify()``.

Supported formats
-----------------
- **miniSEED** (.mseed, .seed) — CTBTO/IMS, USGS, IRIS standard.
  Requires ``obspy`` (optional dependency).
- **SAC** (.sac) — Seismic Analysis Code.  Requires ``obspy``.
- **WAV** (.wav) — any sample rate, any bit depth.
  Uses ``torchaudio`` (already a dependency).
- **NumPy** (.npy, .npz) — raw arrays with sr passed explicitly.
- **Raw binary** — flat float32/float64/int16/int24/int32 with
  sample rate passed explicitly.

All loaders preserve the native dtype of the data.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_mseed(
    path: str | Path,
    *,
    channel: str | None = None,
    merge: bool = True,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Load a miniSEED file (CTBTO/IMS, USGS, IRIS).

    Parameters
    ----------
    path : str or Path
        Path to ``.mseed`` or ``.seed`` file.
    channel : str, optional
        SEED channel code to extract (e.g. ``"BDF"``, ``"BHZ"``).
        If None, uses the first channel found.
    merge : bool
        Merge traces with matching IDs (fills gaps with zeros).
    device : str or torch.device
        Target device.

    Returns
    -------
    (data, sr, meta)
        data : Tensor (n_samples,) — preserves native dtype.
        sr : int — sample rate.
        meta : dict — network, station, location, channel, starttime, etc.
    """
    try:
        import obspy
    except ImportError as e:
        raise ImportError(
            "obspy is required for miniSEED loading. "
            "Install with: pip install obspy") from e

    st = obspy.read(str(path))
    if merge:
        st.merge(fill_value=0)

    if channel is not None:
        st = st.select(channel=channel)
        if len(st) == 0:
            raise ValueError(
                f"No traces with channel={channel!r} in {path}")

    tr = st[0]
    sr = int(tr.stats.sampling_rate)
    arr = tr.data  # numpy array, native dtype

    meta = {
        "network": tr.stats.network,
        "station": tr.stats.station,
        "location": tr.stats.location,
        "channel": tr.stats.channel,
        "starttime": str(tr.stats.starttime),
        "endtime": str(tr.stats.endtime),
        "npts": tr.stats.npts,
        "format": "mseed",
        "source_path": str(path),
    }

    data = torch.from_numpy(arr.copy()).to(device=device)
    return data, sr, meta


def load_sac(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Load a SAC (Seismic Analysis Code) file.

    Parameters
    ----------
    path : str or Path
        Path to ``.sac`` file.
    device : str or torch.device
        Target device.

    Returns
    -------
    (data, sr, meta)
    """
    try:
        import obspy
    except ImportError as e:
        raise ImportError(
            "obspy is required for SAC loading. "
            "Install with: pip install obspy") from e

    st = obspy.read(str(path), format="SAC")
    tr = st[0]
    sr = int(tr.stats.sampling_rate)
    arr = tr.data

    meta = {
        "station": tr.stats.station,
        "channel": tr.stats.channel,
        "starttime": str(tr.stats.starttime),
        "delta": tr.stats.delta,
        "npts": tr.stats.npts,
        "format": "sac",
        "source_path": str(path),
    }

    # Include SAC-specific headers if available
    sac_hdr = getattr(tr.stats, "sac", None)
    if sac_hdr is not None:
        for key in ("stla", "stlo", "stel", "evla", "evlo", "evdp",
                     "mag", "dist", "az", "baz"):
            val = getattr(sac_hdr, key, None)
            if val is not None and val != -12345.0:
                meta[f"sac_{key}"] = float(val)

    data = torch.from_numpy(arr.copy()).to(device=device)
    return data, sr, meta


def load_wav(
    path: str | Path,
    *,
    channel: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Load a WAV file at any sample rate / bit depth.

    Uses torchaudio which handles 8/16/24/32-bit int and 32/64-bit float.

    Parameters
    ----------
    path : str or Path
        Path to ``.wav`` file.
    channel : int
        Channel index to extract (0-based).  Default 0 (mono/left).
    device : str or torch.device
        Target device.

    Returns
    -------
    (data, sr, meta)
    """
    import torchaudio

    info = torchaudio.info(str(path))
    waveform, sr = torchaudio.load(str(path))
    # waveform: (n_channels, n_samples) float32

    if channel >= waveform.shape[0]:
        raise ValueError(
            f"channel={channel} but file has {waveform.shape[0]} channels")

    data = waveform[channel].to(device=device)

    meta = {
        "n_channels": int(info.num_channels),
        "bits_per_sample": int(info.bits_per_sample),
        "encoding": str(info.encoding),
        "n_frames": int(info.num_frames),
        "format": "wav",
        "source_path": str(path),
    }
    return data, int(sr), meta


def load_numpy(
    path: str | Path,
    sr: int,
    *,
    key: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Load a NumPy ``.npy`` or ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Path to ``.npy`` or ``.npz`` file.
    sr : int
        Sample rate (must be provided — not stored in the file).
    key : str, optional
        For ``.npz`` files, the array key to load.
        If None, uses the first key.
    device : str or torch.device
        Target device.

    Returns
    -------
    (data, sr, meta)
    """
    p = Path(path)
    if p.suffix == ".npz":
        npz = np.load(str(p))
        if key is None:
            key = list(npz.files)[0]
        arr = npz[key]
        meta_keys = list(npz.files)
    else:
        arr = np.load(str(p))
        meta_keys = []

    meta = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "format": "numpy",
        "source_path": str(path),
        "npz_keys": meta_keys,
    }

    data = torch.from_numpy(arr.copy()).to(device=device)
    if data.ndim > 1:
        data = data.reshape(-1)  # flatten to 1-D

    return data, sr, meta


def load_raw(
    path: str | Path,
    sr: int,
    dtype: str = "float32",
    *,
    byte_order: str = "little",
    header_bytes: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Load raw binary data (no header format).

    Parameters
    ----------
    path : str or Path
        Path to binary file.
    sr : int
        Sample rate.
    dtype : str
        Sample dtype: ``"float32"``, ``"float64"``, ``"int16"``,
        ``"int24"``, ``"int32"``.
    byte_order : str
        ``"little"`` or ``"big"``.
    header_bytes : int
        Number of bytes to skip at the start of the file.
    device : str or torch.device
        Target device.

    Returns
    -------
    (data, sr, meta)
    """
    p = Path(path)
    raw = p.read_bytes()[header_bytes:]

    if dtype == "int24":
        # 24-bit PCM — 3 bytes per sample, pack into int32
        n_samples = len(raw) // 3
        bo = "<" if byte_order == "little" else ">"
        samples = []
        for i in range(n_samples):
            b = raw[i * 3:(i + 1) * 3]
            if byte_order == "little":
                val = b[0] | (b[1] << 8) | (b[2] << 16)
            else:
                val = (b[0] << 16) | (b[1] << 8) | b[2]
            if val & 0x800000:
                val -= 0x1000000
            samples.append(val)
        arr = np.array(samples, dtype=np.int32)
    else:
        np_dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int16": np.int16,
            "int32": np.int32,
        }
        if dtype not in np_dtype_map:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. "
                f"Use one of: {sorted(np_dtype_map)}")
        np_dt = np_dtype_map[dtype]
        if byte_order == "big":
            np_dt = np.dtype(np_dt).newbyteorder(">")
        arr = np.frombuffer(raw, dtype=np_dt)

    meta = {
        "dtype": dtype,
        "byte_order": byte_order,
        "n_samples": len(arr),
        "header_bytes": header_bytes,
        "format": "raw",
        "source_path": str(path),
    }

    data = torch.from_numpy(arr.copy()).to(device=device)
    return data, sr, meta


def load(
    path: str | Path,
    sr: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    """Auto-detect format and load.

    For formats that don't embed sample rate (numpy, raw), ``sr``
    must be provided.

    Parameters
    ----------
    path : str or Path
        Data file path.
    sr : int, optional
        Sample rate override / requirement.
    **kwargs
        Forwarded to the format-specific loader.

    Returns
    -------
    (data, sr, meta)
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".mseed", ".seed"):
        data, file_sr, meta = load_mseed(p, **kwargs)
    elif suffix == ".sac":
        data, file_sr, meta = load_sac(p, **kwargs)
    elif suffix == ".wav":
        data, file_sr, meta = load_wav(p, **kwargs)
    elif suffix == ".npz":
        if sr is None:
            raise ValueError("sr is required for .npz files")
        data, file_sr, meta = load_numpy(p, sr, **kwargs)
    elif suffix == ".npy":
        if sr is None:
            raise ValueError("sr is required for .npy files")
        data, file_sr, meta = load_numpy(p, sr, **kwargs)
    else:
        if sr is None:
            raise ValueError(f"sr is required for raw files ({suffix})")
        data, file_sr, meta = load_raw(p, sr, **kwargs)

    # Allow sr override
    if sr is not None and sr != file_sr:
        meta["original_sr"] = file_sr
        file_sr = sr

    return data, file_sr, meta
