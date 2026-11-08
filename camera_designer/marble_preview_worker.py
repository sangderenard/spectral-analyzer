"""Bounded, progressive Spectral photograph of one saved Studio mesh.

The caller owns this process. Terminating it cancels a stale design revision;
the worker never runs inside an HTTP request thread. See the marble preview
handoff for the NPZ and JSON input contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    # Windows readers can briefly deny delete sharing on status.json. Keep
    # the complete temporary file and retry the atomic publication.
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.01)


def _write_orbit_avi(output: Path, frames: list[dict], resolution: int,
                     writer_root: Path, on_frame=None) -> Path:
    """Use Turing's JPEG encoder and indexed MJPEG/AVI writer, one frame at a time."""
    sys.path.insert(0, str(writer_root.resolve()))
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.compression.containers.avi import MJPEGAVIWriter
    from src.common.tensors.compression.jpeg.frame import encode_jfif

    target = output / "orbit.avi"
    temporary = output / "orbit.avi.tmp"
    with MJPEGAVIWriter(temporary, width=resolution, height=resolution, fps=8) as writer:
        for index, frame in enumerate(frames, 1):
            pixels = np.asarray(Image.open(output / frame["image"]).convert("RGB"), dtype=np.uint8)
            writer.append_frame(encode_jfif(AbstractTensor.get_tensor(pixels)))
            if on_frame is not None:
                on_frame(index)
    os.replace(temporary, target)
    return target


def _snapshot(accumulator, output: Path, digest: str, epoch: int,
              elapsed: float, preparation: float, rays: int, sensor_sites: int,
              finished: bool = False, transport: str = "fixed:1", *,
              frame_index: int = 0, total_frames: int = 1,
              frame_elapsed: float = 0.0, completed_frames: tuple[dict, ...] = ()) -> dict:
    from OpenGL.GL import GL_TEXTURE_2D, GL_RGBA, GL_FLOAT, glBindTexture, glGetTexImage, glFinish

    glFinish()
    if transport == "fixed:3":
        # FilmStack's ordered R/G/B emulsion layers are separate linear
        # RGBA32F exposure textures. Their RGB components repeat the scalar
        # exposure; read one component from each, preserving relative energy.
        layers = []
        for texture in accumulator._textures[:3]:
            glBindTexture(GL_TEXTURE_2D, texture)
            raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
            layers.append(np.frombuffer(raw, dtype=np.float32).reshape(
                accumulator._h, accumulator._w, 4)[:, :, 0].copy())
        linear = np.stack(layers, axis=2)
        luma = np.sum(linear * [0.2126, 0.7152, 0.0722], axis=2)
        nonzero = luma[luma > 0]
        white = max(float(np.percentile(nonzero, 99)) if len(nonzero) else 0.0, 1e-6)
        mapped = 1.0 - np.exp(-np.maximum(linear, 0.0) * (2.0 / white))
        display = np.where(mapped <= 0.0031308, 12.92 * mapped,
                           1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)
        display = np.uint8(np.flipud(np.clip(display, 0, 1)) * 255)
        mode = "RGB"
        hdr_name = f"exposure-{epoch:06d}.npz" if finished else None
        if finished:
            temporary_hdr = output / (hdr_name + ".tmp")
            with temporary_hdr.open("wb") as stream:
                np.savez_compressed(stream, linear_rgb_exposure=np.flipud(linear),
                                    wavelengths_nm=np.asarray([650, 550, 450], np.float32))
            os.replace(temporary_hdr, output / hdr_name)
    else:
        glBindTexture(GL_TEXTURE_2D, accumulator.tex)
        raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(raw, dtype=np.float32).reshape(
            accumulator._h, accumulator._w, 4).copy()
        luma = image[:, :, :3].mean(axis=2)
        nonzero = luma[luma > 0]
        white = max(float(np.percentile(nonzero, 98)) if len(nonzero) else 0.0, 1e-6)
        display = np.uint8(np.flipud(np.clip(luma / white, 0, 1)) * 255)
        mode = "L"
        hdr_name = None
    filename = f"preview-{epoch:06d}.png"
    temporary = output / (filename + ".tmp")
    Image.fromarray(display, mode=mode).save(temporary, format="PNG")
    os.replace(temporary, output / filename)
    frames = [*completed_frames]
    if finished:
        frames.append({"index": frame_index, "image": filename, "hdr": hdr_name})
    status = {
        "digest": digest,
        "state": ("encoding" if finished and frame_index + 1 == total_frames and total_frames > 1 else
                  "complete" if finished and frame_index + 1 == total_frames else "exposing"),
        "image": filename,
        "frames": frames,
        "frame_index": frame_index,
        "total_frames": total_frames,
        "epoch": epoch,
        "elapsed_seconds": round(elapsed, 3),
        "frame_elapsed_seconds": round(frame_elapsed, 3),
        "preparation_seconds": round(preparation, 3),
        "forward_ray_jobs": rays,
        "sensor_sites": sensor_sites,
        "visible_pixels": int(np.count_nonzero(luma)),
        "display_white": white,
        "transport": ("fixed:3@450,550,650nm" if transport == "fixed:3"
                      else "fixed:1@587.5618nm"),
        "camera": "Spectral analytical thin lens",
    }
    if hdr_name:
        status["hdr"] = hdr_name
        status["hdr_channels"] = "linear sensor exposure, ordered R/G/B; common display tonemap"
    _atomic_json(output / "status.json", status)
    return status


def run(payload: Path, config_path: Path, output: Path, *,
        seconds: float = 600.0, snapshot_seconds: float = 2.0,
        max_epochs: int | None = None, avi_writer_root: Path | None = None) -> dict:
    import demo_pluck_gl as renderer
    from camera_designer.camera_preset import simple_doublet_preset
    from camera_designer.scene_builder import (
        append_saved_studio_mesh_to_gpu_scene, build_gpu_scene,
    )

    if not 0 < seconds <= 3600:
        raise ValueError("photo exposure per frame must be in (0, 3600] seconds")
    if not 0.25 <= snapshot_seconds <= 60:
        raise ValueError("snapshot interval must be in [0.25, 60] seconds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    total_frames = int(config.get("orbit_frames", 1))
    if total_frames not in (1, 8, 16, 32):
        raise ValueError("orbit_frames must be 1, 8, 16, or 32")
    transport = str(config.get("transport", "fixed:1"))
    if transport not in ("fixed:1", "fixed:3"):
        raise ValueError("transport must be fixed:1 or fixed:3")
    wavelengths = [.5875618] if transport == "fixed:1" else [.450, .550, .650]
    digest = hashlib.sha256(payload.read_bytes() +
                            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "status.json", {
        "digest": digest, "state": "preparing", "image": None,
        "frames": [], "frame_index": 0, "total_frames": total_frames,
        "transport": ("fixed:3@450,550,650nm" if transport == "fixed:3"
                      else "fixed:1@587.5618nm"),
    })
    with np.load(payload, allow_pickle=False) as saved:
        vertices = np.asarray(saved["vertices"], np.float32)
        material_ids = np.asarray(saved["material_ids"], np.int32)
        pbr = np.asarray(saved["pbr"], np.float32)

    preparation_start = time.monotonic()
    renderer._ensure_worker_gl_context()
    preset = simple_doublet_preset()
    # The analytical sensor lens forms this image. The demo's physical
    # doublet is behind the virtual eye and would only add unused BVH work.
    preset.lens_group.elements = []
    checker = config.get("checker", {})
    checker_pos = checker.get("center_m", [0, 0, .30])
    checker_width = float(checker.get("width_m", .15))
    lights = config.get("lights", [
        {"pos": [.14, .10, .17], "dir": [-.14, -.10, .13],
         "power": 1.0, "label": "preview-stage"},
    ])
    scene = build_gpu_scene(
        preset, lights,
        scene_object={"type": "checker_target", "checks": int(checker.get("checks", 4)),
                      "width": checker_width, "pos": checker_pos, "emissive": True,
                      "normal_sign": int(checker.get("normal_sign", -1))},
        wavelengths_um=wavelengths,
    )
    scene = append_saved_studio_mesh_to_gpu_scene(
        scene, vertices, material_ids, pbr,
        scale_m=float(config.get("scale_m", .025)),
        center_m=config.get("object_center_m", [0, 0, .18]),
        wavelengths_um=wavelengths,
    )
    geom, shade, mat_buf, triangles, contexts, _, bounds, sources = scene
    margin = np.asarray([.02, .02, .04], np.float32)
    bounds = (bounds[0] - margin, bounds[1] + margin)
    nodes, ids = renderer._build_gpu_bvh(triangles)
    film = renderer.FILM_STACKS["em_rgb"] if transport == "fixed:3" else None
    state = renderer._gpu_ray_field_prebuilt(
        geom, shade, mat_buf, nodes, ids, contexts, sources, bounds,
        dims=(96, 96, 96), dispatch_batch=512, max_bounces=4, initial_rays=128,
        film_stack=film,
    )
    if state is None:
        raise RuntimeError("Spectral GPU ray-field shader did not link")
    camera = config.get("camera", {})
    resolution = int(camera.get("resolution", 256))
    if resolution not in (128, 256, 512, 768, 1024, 2048):
        raise ValueError("preview resolution must be 128, 256, 512, 768, 1024, or 2048")
    eye = np.asarray(camera.get("eye_m", [0, 0, .07]), np.float32)
    target = np.asarray(camera.get("target_m", checker_pos), np.float32)
    def sensor(frame_index: int):
        angle = 2.0 * np.pi * frame_index / total_frames
        dx, dy, dz = eye - target
        camera_eye = target + np.asarray((np.cos(angle) * dx + np.sin(angle) * dz,
                                          dy, -np.sin(angle) * dx + np.cos(angle) * dz), np.float32)
        accumulator = renderer.SensorAccumulator(
            ssbo_tris=state["ssbo"][0], ssbo_nodes=state["ssbo"][1],
            ssbo_ids=state["ssbo"][2], ssbo_shade=state["ssbo"][9],
            ssbo_mat=state["ssbo"][10], n_tris=len(triangles),
            n_bvh_nodes=len(nodes), tex_bands=state["tex_bands"],
            bmin=state["bmin"], bmax=state["bmax"], dims=state["dims"],
            camera_eye=camera_eye, camera_target=target,
            sensor_w=resolution, sensor_h=resolution, rows_per_frame=16,
            fov_deg=float(camera.get("fov_degrees", 20)), max_bounces=4,
            aperture_radius=float(camera.get("aperture_radius_m", .005)),
            focus_dist=float(camera.get("focus_distance_m", .23)),
            n_blades=int(camera.get("aperture_blades", 0)),
            aperture_rot=float(camera.get("aperture_rotation_radians", 0)),
            vol_alpha=0.0, vol_steps=8,
        )
        if not accumulator._active:
            accumulator.destroy()
            raise RuntimeError("Spectral sensor compute shader did not link")
        if film is not None:
            accumulator.film = film
        accumulator.half_life = 0.0
        accumulator._sprinkle = False  # existing deterministic row-priority mode
        return accumulator
    preparation = time.monotonic() - preparation_start
    epoch = 0
    completed_frames: list[dict] = []
    accumulator = None
    try:
        from OpenGL.GL import glFinish
        job_start = time.monotonic()
        for frame_index in range(total_frames):
            accumulator = sensor(frame_index)
            frame_start = time.monotonic()
            last_snapshot = frame_start - snapshot_seconds
            frame_epochs = 0
            while True:
                renderer._gpu_pump_prebuilt(state, n_rays=8192)
                accumulator.tick(strips=4)
                epoch += 1
                frame_epochs += 1
                if epoch % 8 == 0:
                    # Do not queue an unbounded number of epochs ahead of the GPU.
                    glFinish()
                if epoch % 4096 == 0:
                    # R32UI forward voxels must not overflow during long exposures.
                    glFinish()
                    seed = state["_pump_seed"]
                    renderer._clear_prebuilt(state)
                    state["_pump_seed"] = seed
                now = time.monotonic()
                done = (now - frame_start >= seconds or
                        (max_epochs is not None and frame_epochs >= max_epochs))
                if now - last_snapshot >= snapshot_seconds or done:
                    status = _snapshot(accumulator, output, digest, epoch,
                                       now - job_start, preparation,
                                       epoch * 8192, frame_epochs * 4 * 16 * resolution,
                                       finished=done, transport=transport,
                                       frame_index=frame_index, total_frames=total_frames,
                                       frame_elapsed=now - frame_start,
                                       completed_frames=tuple(completed_frames))
                    print(json.dumps(status, sort_keys=True), flush=True)
                    last_snapshot = now
                if done:
                    completed_frames = status["frames"]
                    accumulator.destroy()
                    accumulator = None
                    break
        if total_frames > 1:
            if avi_writer_root is None:
                raise ValueError("orbit animation needs the Turing AVI writer root")
            encoding_start = time.monotonic()
            def encoding_progress(frame_count: int) -> None:
                progress = {**status, "state": "encoding", "encoding_frames": frame_count,
                            "encoding_elapsed_seconds": round(time.monotonic() - encoding_start, 3)}
                _atomic_json(output / "status.json", progress)
                print(json.dumps({"state": "encoding", "encoding_frames": frame_count,
                                  "total_frames": total_frames,
                                  "encoding_elapsed_seconds": progress["encoding_elapsed_seconds"]}), flush=True)
            _write_orbit_avi(output, completed_frames, resolution, avi_writer_root,
                             on_frame=encoding_progress)
            status = {**status, "state": "complete", "encoding_frames": total_frames,
                      "encoding_elapsed_seconds": round(time.monotonic() - encoding_start, 3),
                      "animation": "orbit.avi",
                      "animation_expires_at": time.time() + 24 * 3600}
            _atomic_json(output / "status.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
        return status
    finally:
        if accumulator is not None:
            accumulator.destroy()
        renderer._destroy_prebuilt(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--snapshot-seconds", type=float, default=2.0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--native-module-dir", type=Path)
    parser.add_argument("--avi-writer-root", type=Path)
    args = parser.parse_args()
    if args.native_module_dir:
        sys.path.append(str(args.native_module_dir.resolve()))
    try:
        run(args.payload, args.config, args.out, seconds=args.seconds,
            snapshot_seconds=args.snapshot_seconds, max_epochs=args.max_epochs,
            avi_writer_root=args.avi_writer_root)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.out / "status.json", {"state": "error", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
