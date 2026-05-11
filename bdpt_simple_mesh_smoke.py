from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import _spectral_kernels as _sk
from ray_tracer_bridge import per_tri_spectral_to_mat_buf


def _tri_normal(tri: np.ndarray) -> np.ndarray:
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    n = np.cross(e1, e2)
    nn = np.linalg.norm(n)
    if nn <= 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / nn


def _rot_x(a: float) -> np.ndarray:
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _save_rgb(path: Path, rgb01: np.ndarray) -> None:
    arr = np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr, mode="RGB").save(path)
        return
    except Exception:
        pass

    ppm_path = path.with_suffix(".ppm")
    h, w, _ = arr.shape
    with ppm_path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())
    print(f"[warn] PIL unavailable, wrote {ppm_path}")


def _aggregate_pixel_cone_records(
    recs: np.ndarray,
    *,
    n_bands: int,
    width: int,
    height: int,
    sensor_group_id: int,
) -> np.ndarray:
    recs = np.ascontiguousarray(recs, dtype=np.float32)
    if recs.size == 0:
        return np.zeros((n_bands, height, width), dtype=np.float32)

    cols_i32 = recs.view(np.int32).reshape(-1, 16)
    group_ids = cols_i32[:, 2]
    band_ids = cols_i32[:, 1].astype(np.int64)
    subpath_ids = cols_i32[:, 0].astype(np.int64)
    vertex_idx = cols_i32[:, 3]

    keep = (
        (group_ids == int(sensor_group_id))
        & (vertex_idx >= 0)
        & (band_ids >= 0)
        & (band_ids < int(n_bands))
        & (subpath_ids >= 0)
        & (subpath_ids < int(width * height))
    )
    if not np.any(keep):
        return np.zeros((n_bands, height, width), dtype=np.float32)

    amp_re = recs[:, 12].astype(np.float64)
    amp_im = recs[:, 13].astype(np.float64)
    amp = np.sqrt(amp_re * amp_re + amp_im * amp_im)

    flat = band_ids[keep] * int(width * height) + subpath_ids[keep]
    counts = np.bincount(
        flat,
        weights=amp[keep],
        minlength=int(n_bands * width * height),
    )
    return counts.reshape(n_bands, height, width).astype(np.float32, copy=False)


def main() -> int:
    out_path = Path("bdpt_simple_mesh.png")

    w = 128
    h = 96
    n_bands = 8
    n_aperture = 8

    freq_hz = (299_792_458.0 / np.linspace(700e-9, 420e-9, n_bands)).astype(np.float64)
    atmo_abs = np.zeros((n_bands,), dtype=np.float64)

    # Build a simple tilted quad mesh in front of camera (2 triangles).
    quad = np.array(
        [
            [-0.9, -0.6, 2.2],
            [0.9, -0.6, 2.2],
            [0.9, 0.6, 2.2],
            [-0.9, 0.6, 2.2],
        ],
        dtype=np.float64,
    )
    R = _rot_y(math.radians(12.0)) @ _rot_x(math.radians(-8.0))
    quad = (quad @ R.T).astype(np.float64)

    mesh_tris = np.array(
        [
            [quad[0], quad[1], quad[2]],
            [quad[0], quad[2], quad[3]],
        ],
        dtype=np.float64,
    )

    # Sensor plane mesh (2 triangles), centered at z=0.
    sensor_w_m = 0.036
    sensor_h_m = 0.024
    sw = 0.5 * sensor_w_m
    sh = 0.5 * sensor_h_m
    sensor_quad = np.array(
        [
            [-sw, -sh, 0.0],
            [sw, -sh, 0.0],
            [sw, sh, 0.0],
            [-sw, sh, 0.0],
        ],
        dtype=np.float64,
    )
    sensor_tris = np.array(
        [
            [sensor_quad[0], sensor_quad[2], sensor_quad[1]],
            [sensor_quad[0], sensor_quad[3], sensor_quad[2]],
        ],
        dtype=np.float64,
    )

    tris = np.concatenate([mesh_tris, sensor_tris], axis=0)
    n_tri = tris.shape[0]
    verts_flat = tris.reshape(n_tri, 9)
    normals = np.stack([_tri_normal(t) for t in tris], axis=0).astype(np.float64)

    # Two simple materials: mesh reflective, sensor dark.
    refl_re = np.zeros((n_tri, n_bands), dtype=np.float64)
    refl_im = np.zeros((n_tri, n_bands), dtype=np.float64)
    diffusion = np.zeros((n_tri, n_bands), dtype=np.float64)

    refl_re[0:2, :] = 0.82
    diffusion[0:2, :] = 0.08

    refl_re[2:4, :] = 0.02
    diffusion[2:4, :] = 0.0

    mat_idx, mat_buf, mat_n_mats = per_tri_spectral_to_mat_buf(
        refl_re,
        refl_im,
        diffusion,
        freq_hz,
    )

    tracer = _sk.RayTracer(
        n_tri=n_tri,
        verts=np.ascontiguousarray(verts_flat, dtype=np.float64),
        normals=np.ascontiguousarray(normals, dtype=np.float64),
        mat_idx=np.ascontiguousarray(mat_idx, dtype=np.int32),
        mat_buf=np.ascontiguousarray(mat_buf, dtype=np.float32),
        mat_n_mats=int(mat_n_mats),
        freq_hz=np.ascontiguousarray(freq_hz, dtype=np.float64),
        speed_m_s=float(299_792_458.0),
        atmo_abs=np.ascontiguousarray(atmo_abs, dtype=np.float64),
    )

    tracer.clear_tri_groups()

    tri_role_emissive = int(getattr(_sk, "TRI_GROUP_ROLE_EMISSIVE", 1 << 0))
    tri_role_sensor = int(getattr(_sk, "TRI_GROUP_ROLE_SENSOR", 1 << 1))
    tri_sample_area = int(getattr(_sk, "TRI_GROUP_SAMPLE_AREA", 1))
    tri_sample_pixel = int(getattr(_sk, "TRI_GROUP_SAMPLE_PIXEL_CONE", 3))

    emitter_group_id = tracer.register_tri_group(
        role_bits=tri_role_emissive,
        sample_policy=tri_sample_area,
        tri_indices=np.asarray([0, 1], dtype=np.int32),
    )

    sensor_group_id = tracer.register_tri_group(
        role_bits=tri_role_sensor,
        sample_policy=tri_sample_pixel,
        tri_indices=np.asarray([2, 3], dtype=np.int32),
        sensor_camera={
            "pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "fwd": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "up": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "sensor_w_m": float(sensor_w_m),
            "sensor_h_m": float(sensor_h_m),
            "focal_m": 0.050,
            "aperture_radius_m": 0.006,
            "n_px": int(w),
            "n_py": int(h),
            "n_aperture_samples": int(n_aperture),
            "pixel_stream_divisor": 1,
            "pixel_stream_phase": 0,
            "pixel_stream_phase_from_seed": 1,
            "aperture_stop_group_id": -1,
        },
    )

    if emitter_group_id < 0 or sensor_group_id < 0:
        raise RuntimeError("Failed to register BDPT tri groups")

    n_rays_per_emitter = np.asarray([2048], dtype=np.int32)
    emitter_max = int(np.sum(n_rays_per_emitter)) * 2 * n_bands
    sensor_max = int(w) * int(h) * int(n_aperture) * int(n_bands)
    max_records = int(emitter_max + sensor_max + 4096)

    recs = tracer.bidirectional_packed(
        n_rays_per_emitter=n_rays_per_emitter,
        max_bounces=1,
        min_amplitude=1.0e-4,
        seed=7,
        max_records=max_records,
    )

    recs = np.ascontiguousarray(recs, dtype=np.float32)
    print(f"records: {recs.shape[0]} rows")

    img_mag = _aggregate_pixel_cone_records(
        recs,
        n_bands=n_bands,
        width=w,
        height=h,
        sensor_group_id=int(sensor_group_id),
    )

    # Simple band-to-RGB slicing for smoke-visualization.
    r = img_mag[min(n_bands - 1, int(n_bands * 0.80)), :, :]
    g = img_mag[min(n_bands - 1, int(n_bands * 0.50)), :, :]
    b = img_mag[min(n_bands - 1, int(n_bands * 0.20)), :, :]
    rgb = np.stack([r, g, b], axis=-1)

    p = float(np.percentile(rgb, 99.0)) if rgb.size else 0.0
    p = max(p, 1.0e-8)
    rgb = np.clip(np.log1p((rgb / p) * 8.0) / math.log1p(8.0), 0.0, 1.0)

    _save_rgb(out_path, rgb)
    print(f"wrote {out_path}")

    nonzero = int(np.count_nonzero(rgb > 0.0))
    lit_pixels = int(np.count_nonzero(np.sum(img_mag, axis=0) > 0.0))
    print(f"nonzero_rgb_pixels: {nonzero}")
    print(f"lit_sensor_pixels: {lit_pixels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
