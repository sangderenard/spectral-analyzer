"""Bounded tile plan and disk-backed assembly for the existing Studio worker.

Tiles are sensor windows, not cameras. PNG/HDR output is assembled from linear
sample means, with one full-image white reference and the existing color law.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import struct
import zlib


def tile_plan(config):
    camera = config.get("camera", {})
    resolution = camera.get("resolution", 256)
    oversampling = config.get("oversampling", 1)
    if type(oversampling) is not int or not 1 <= oversampling <= 4:
        raise ValueError("oversampling must be an integer multiplier from 1 to 4")
    custom = "width" in camera or "height" in camera
    if "tiled" not in config and not custom and oversampling == 1 and resolution in (128,256,512,768,1024,2048,4096):
        return None
    if custom and not ("width" in camera and "height" in camera):
        raise ValueError("custom camera size requires both width and height")
    value = config.get("tiled", {"tile_size": 1024})
    if not isinstance(value, dict) or set(value) != {"tile_size"}:
        raise ValueError("tiled must contain only tile_size")
    size = value["tile_size"]
    if type(size) is not int or not 128 <= size <= 4096:
        raise ValueError("tiled.tile_size must be an integer from 128 to 4096")
    width, height = camera.get("width", resolution), camera.get("height", resolution)
    if any(type(edge) is not int or not 128 <= edge <= 32768 for edge in (width, height)):
        raise ValueError("tiled image dimensions must be integers from 128 to 32768")
    if (config.get("transport", "fixed:1") != "fixed:3" or config.get("orbit_frames", 1) != 1
            or "live" in config or config.get("client_intersections", False)):
        raise ValueError("Tiled photos require RGB transport, one still frame, and server intersections")
    output_width, output_height = width, height
    width, height = width * oversampling, height * oversampling
    if width > 32768 or height > 32768:
        raise ValueError("oversampled sensor dimensions exceed 32768")
    count = math.ceil(width / size) * math.ceil(height / size)
    if count > 4096:
        raise ValueError("tiled photo exceeds 4096 tiles; increase tile_size")
    return {"version": 1, "width": width, "height": height,
            "output_width": output_width, "output_height": output_height, "oversampling": oversampling, "tile_size": size,
            "total_tiles": count, "completed_tiles": 0, "active_tile": None,
            "exposure_seconds_per_tile": float(config.get("exposure_seconds", 30)),
            "requested_exposure_seconds": count * float(config.get("exposure_seconds", 30)),
            "required_disk_bytes": width * height * 100 + 64 * 1024 * 1024,
            "coordinate_origin": "top-left", "execution": "serial sensor windows in the owned photo worker"}


def tile_windows(plan):
    size = plan["tile_size"]
    for y in range(0, plan["height"], size):
        for x in range(0, plan["width"], size):
            yield {"x": x, "y": y, "width": min(size, plan["width"] - x),
                   "height": min(size, plan["height"] - y)}


def require_tile_disk(output, plan):
    free = shutil.disk_usage(output).free
    if free < plan["required_disk_bytes"]:
        raise ValueError(f"Tiled photo needs {plan['required_disk_bytes']} free disk bytes; {free} available")


def _png_chunk(stream, kind, data):
    stream.write(struct.pack(">I", len(data)))
    stream.write(kind)
    stream.write(data)
    stream.write(struct.pack(">I", zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))


def write_rgb_png(path, width, height, rows):
    """Write bounded row batches without allocating a full-size display image."""
    with Path(path).open("wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        _png_chunk(stream, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        _png_chunk(stream, b"sRGB", b"\x00")
        compressor = zlib.compressobj()
        count = 0
        for batch in rows:
            for row in batch:
                packed = compressor.compress(b"\x00" + row.tobytes())
                if packed:
                    _png_chunk(stream, b"IDAT", packed)
                count += 1
        if count != height:
            raise ValueError("PNG row count does not match the image height")
        _png_chunk(stream, b"IDAT", compressor.flush())
        _png_chunk(stream, b"IEND", b"")


class TileAssembly:
    def __init__(self, output, plan):
        import numpy as np
        self.output, self.plan = Path(output), plan
        shape = (plan["height"], plan["width"], 3)
        self.raw = np.lib.format.open_memmap(self.output / "tile-sensor.npy", mode="w+", dtype="<f4", shape=shape)
        self.scratch = []
        self.completed = 0
        self.samples = []

    def add(self, window, hdr_path, samples):
        import numpy as np
        if samples < 1:
            raise ValueError("A tile must finish at least one complete sensor pass")
        expected = next(w for i, w in enumerate(tile_windows(self.plan)) if i == self.completed)
        if window != expected:
            raise ValueError("Tile is not the next queued sensor window")
        with np.load(hdr_path, allow_pickle=False) as data:
            raw = data["linear_rgb_exposure"]
        if raw.shape != (window["height"], window["width"], 3) or not np.isfinite(raw).all():
            raise ValueError("Tile HDR has invalid shape or nonfinite samples")
        x, y = window["x"], window["y"]
        self.raw[y:y+window["height"], x:x+window["width"]] = raw / samples
        self.raw.flush()
        self.completed += 1
        self.samples.append(samples)

    def finish(self, response, gamma, geometry_stats=None):
        import json
        import numpy as np
        from camera_designer.marble_color_profile import sensor_to_linear_srgb, render_linear_srgb, SRGB_LUMA, display_transform_metadata
        from camera_designer.film_response import normalize_film_response
        if self.completed != self.plan["total_tiles"]:
            raise ValueError("Cannot assemble an incomplete tile queue")
        width, height = self.plan["output_width"], self.plan["output_height"]
        factor = self.plan["oversampling"]
        final_raw = self.raw
        if factor > 1:
            final_raw = np.lib.format.open_memmap(self.output / "tile-output.npy", mode="w+", dtype="<f4", shape=(height,width,3))
            self.scratch.append(final_raw)
            for y in range(0, height, 32):
                end = min(y+32,height)
                block = self.raw[y*factor:end*factor]
                final_raw[y:end] = block.reshape(end-y,factor,width,factor,3).mean(axis=(1,3),dtype=np.float64)
        rgb = np.lib.format.open_memmap(self.output / "tile-srgb.npy", mode="w+", dtype="<f8", shape=final_raw.shape)
        luminance = np.memmap(self.output / "tile-luma.bin", mode="w+", dtype="<f8", shape=(width*height,))
        self.scratch.extend((rgb, luminance))
        n = 0
        profile = None
        for y in range(0, height, 32):
            rgb[y:y+32], profile = sensor_to_linear_srgb(final_raw[y:y+32], response)
            lum = np.maximum(rgb[y:y+32] @ SRGB_LUMA, 0.0)
            positive = lum[lum > 0]
            luminance[n:n+len(positive)] = positive
            n += len(positive)
        # Exact NumPy linear percentile, partitioned in the disk-backed array.
        # No full-frame ndarray/copy and no per-tile normalization seams.
        white = 1e-6
        if n:
            k = (n - 1) * .99
            lo, hi = math.floor(k), math.ceil(k)
            luminance[:n].partition((lo, hi))
            white = max(float(luminance[lo] + (luminance[hi] - luminance[lo]) * (k - lo)), 1e-6)
        luminance._mmap.close()
        self.scratch.pop()
        del luminance
        (self.output / "tile-luma.bin").unlink()
        def rows():
            for y in range(0, height, 32):
                yield render_linear_srgb(rgb[y:y+32], {"gamma": gamma,
                    "normalization": "manual", "normalization_white": white})[0]
        png = self.output / "preview-000000.png"
        write_rgb_png(png.with_suffix(".tmp"), width, height, rows())
        os.replace(png.with_suffix(".tmp"), png)
        hdr = self.output / "exposure-000000.npz"
        response = normalize_film_response() if response is None else normalize_film_response(response)
        with hdr.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, linear_rgb_exposure=final_raw, linear_srgb_exposure=rgb,
                wavelengths_nm=np.asarray([650,550,450],np.float32),
                film_response_version=np.asarray(response["version"],np.int32),
                film_centers_nm=np.asarray(response["centers_nm"],np.float64),
                film_sigmas_oct=np.asarray(response["sigmas_oct"],np.float64),
                color_profile_json=np.asarray(json.dumps(profile,sort_keys=True)),
                geometry_stats_json=np.asarray(json.dumps(geometry_stats,sort_keys=True)),
                tile_samples=np.asarray(self.samples,np.uint64),
                oversampling=np.asarray(factor,np.int32),
                sensor_dimensions=np.asarray([self.plan["width"],self.plan["height"]],np.int32),
                exposure_normalization=np.asarray("mean per complete sensor sample"))
        os.replace(hdr.with_suffix(".tmp"), hdr)
        self.close()
        del rgb
        (self.output / "tile-srgb.npy").unlink()
        (self.output / "tile-sensor.npy").unlink()
        (self.output / "tile-output.npy").unlink(missing_ok=True)
        return {"image": png.name, "hdr": hdr.name, "display_white": white,
                "display_gamma": gamma, "color_profile": profile,
                "visible_pixels": n, "width": width, "height": height,
                "display_transform": display_transform_metadata({"gamma": gamma}, effective_white=white),
                "samples_per_pixel_min": min(self.samples), "samples_per_pixel_max": max(self.samples),
                "hdr_channels": "mean linear sensor samples plus separate profiled linear sRGB; one full-image display transform"}

    def close(self):
        for mapping in self.scratch:
            mapping._mmap.close()
        self.scratch.clear()
        if self.raw is not None:
            self.raw.flush()
            self.raw._mmap.close()
            self.raw = None

