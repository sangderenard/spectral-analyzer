"""Bounded, progressive Spectral photograph of one saved Studio mesh.

The caller owns this process. Terminating it cancels a stale design revision;
the worker never runs inside an HTTP request thread. See the marble preview
handoff for the NPZ and JSON input contract.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from camera_designer.film_response import build_film_stack, normalize_film_response
from camera_designer.marble_color_profile import (
    sensor_to_linear_srgb, render_linear_srgb, display_transform_metadata, display_white_reference, SRGB_LUMA)
from camera_designer.studio_spectral_display import project_spectral_exposure

MAX_MARBLE_BOUNCES = 200


def _sensor_device_limits(width, height):
    """Actual device limits plus minimum sensor storage, not a total VRAM promise."""
    from OpenGL import GL
    maximum=int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
    if width>maximum or height>maximum:
        raise ValueError(f'Sensor {width}x{height} exceeds device texture limit {maximum}')
    result={'max_texture_size':maximum,'sensor_texture_bytes':int(width)*int(height)*48,
            'available_gpu_memory_bytes':None,'memory_report':'unavailable'}
    try:
        extensions={GL.glGetStringi(GL.GL_EXTENSIONS,i) for i in range(int(GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)))}
        if b'GL_NVX_gpu_memory_info' in extensions:
            free=int(GL.glGetIntegerv(0x9049))*1024
            if free>0:result.update(available_gpu_memory_bytes=free,memory_report='driver NVX current available; advisory')
    except Exception:pass
    return result


def _bounce_limit(value=4) -> int:
    """Transport depth, never a display effect or a hidden lower clamp."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_bounces must be an integer in [1, 200]")
    try:
        numeric = float(value)
    except (ValueError, OverflowError):
        raise ValueError("max_bounces must be an integer in [1, 200]") from None
    if not np.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= MAX_MARBLE_BOUNCES:
        raise ValueError("max_bounces must be an integer in [1, 200]")
    return int(numeric)


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


def _camera_pose_sequence(config):
    """Read server-resolved cameras; lens/back derivation belongs to admission."""
    if "camera_poses" not in config:
        return None
    poses = config["camera_poses"]
    if not isinstance(poses, list) or not 1 <= len(poses) <= 32:
        raise ValueError("camera_poses must contain 1..32 resolved cameras")
    from camera_designer.marble_tiles import tile_plan
    if (config.get("orbit_frames", 1) != 1 or tile_plan(config) is not None
            or "live" in config or config.get("client_intersections", False)):
        raise ValueError("camera_poses cannot be combined with orbit, tiles, live view or client intersections")
    resolution = config.get("camera", {}).get("resolution", 256)
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict) or "projection" in pose or "width" in pose or "height" in pose:
            raise ValueError(f"camera_poses[{index}] must be a resolved analytical camera")
        if pose.get("resolution") != resolution:
            raise ValueError("camera_poses must share the photo resolution")
        for name in ("eye_m", "target_m"):
            vector = pose.get(name)
            if (not isinstance(vector, list) or len(vector) != 3
                    or any(type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 10 for v in vector)):
                raise ValueError(f"camera_poses[{index}].{name} must contain three finite scene coordinates")
        delta = np.asarray(pose["target_m"], np.float32) - np.asarray(pose["eye_m"], np.float32)
        if float(np.linalg.norm(delta)) <= 1e-6:
            raise ValueError("camera pose eye and target must be distinct in the sensor basis")
        for name in ("fov_degrees", "aperture_radius_m", "focus_distance_m", "aperture_blades", "aperture_rotation_radians"):
            if type(pose.get(name)) not in (int, float) or not np.isfinite(pose[name]):
                raise ValueError(f"camera_poses[{index}].{name} must be resolved and finite")
    return poses


def _sensor_lane_cameras(config):
    """Resolved simultaneous sensors, in stable lane order (main camera first)."""
    lanes = config.get("sensor_lanes")
    if lanes is None:
        return None
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 32:
        raise ValueError("sensor_lanes must contain 1..32 resolved cameras")
    if config.get("camera_poses") is not None or config.get("orbit_frames", 1) != 1:
        raise ValueError("simultaneous sensor lanes cannot be a camera sequence")
    _camera_pose_sequence({"camera": config.get("camera", {}), "camera_poses": lanes})
    if lanes[0] != config.get("camera"):
        raise ValueError("sensor_lanes[0] must be the main camera")
    if any(lane.get('sensor_transport', config.get('transport','fixed:3')) not in ('fixed:1', 'fixed:3', 'fixed:4', 'fixed:8')
           for lane in lanes):
        raise ValueError('sensor lane transport must be fixed:1, fixed:3, fixed:4 or fixed:8')
    if any(lane.get('sensor_output', 'rgb') not in ('rgb', 'mono') for lane in lanes):
        raise ValueError('sensor lane output must be rgb or mono')
    return lanes


def _require_disjoint_lane_roots(scenes):
    """A native forward root may belong to exactly one simultaneous lane."""
    seen = set()
    for index, scene in enumerate(scenes):
        roots = np.asarray(scene[7], np.float32)
        if roots.ndim != 2 or roots.shape[1] != 12:
            raise ValueError("sensor lane has invalid native SourceRec12 roots")
        # The same position/direction/wavelength is the same forward origin,
        # regardless of its beta or material's local triangle ordinal.
        for row in roots:
            key = row[[0, 1, 2, 4, 5, 6, 8]].tobytes()
            if key in seen:
                raise ValueError(f"sensor lane {index} reuses a forward ray origin")
            seen.add(key)


def _dispatch_timed_sensor_lanes(lanes, interval, slice_id, *, prepare_scene,
                                 renderer, barrier, sweeps, finish, geometry_revision=0):
    """Pack one captured slice into every independent field and sensor owner."""
    from camera_designer.studio_native_emitters import install_timed_light_frame
    admitted = []
    for lane in lanes:
        installed_geometry=lane.get('installed_fixture_geometry')
        stale=lane.get('unsupported_geometry_revision')==geometry_revision
        if stale:
            lane['stale_geometry_slices']=lane.get('stale_geometry_slices',0)+1
            try:
                current_scene,current_spheres=prepare_scene(lane['camera'],installed_geometry)
                from camera_designer.studio_native_emitters import timed_light_buffers
                timed_light_buffers(lane['scene'],current_scene)
            except ValueError:
                current_scene=lane.get('installed_scene',lane['scene'])
                current_spheres=lane['installed_spheres']
        else:
            try:
                current_scene,current_spheres=prepare_scene(lane['camera'])
                # Prove that the captured triangle slice can retain the prepared
                # source identity before changing either GPU reader.
                if geometry_revision!=lane.get('installed_geometry_revision',0):
                    from camera_designer.studio_native_emitters import timed_light_buffers
                    timed_light_buffers(lane['scene'],current_scene)
            except ValueError:
                if geometry_revision==lane.get('installed_geometry_revision',0):raise
                lane['unsupported_geometry_revision']=geometry_revision
                lane['stale_geometry_slices']=lane.get('stale_geometry_slices',0)+1
                try:
                    current_scene,current_spheres=prepare_scene(lane['camera'],installed_geometry)
                    from camera_designer.studio_native_emitters import timed_light_buffers
                    timed_light_buffers(lane['scene'],current_scene)
                except ValueError:
                    current_scene=lane.get('installed_scene',lane['scene'])
                    current_spheres=lane['installed_spheres']
        roots = install_timed_light_frame(lane['state'], lane['accumulator'],
                                          lane['scene'], current_scene,
                                          installed_scene=lane.get('installed_scene',lane['scene']),
                                          renderer=renderer,sphere_regions=current_spheres)
        lane['installed_scene']=current_scene
        lane['installed_spheres']=current_spheres
        if not stale and lane.get('unsupported_geometry_revision')!=geometry_revision:
            lane['installed_geometry_revision']=geometry_revision
        seed = lane['state']['_pump_seed']
        renderer._clear_prebuilt(lane['state'])
        lane['state']['_pump_seed'] = seed
        admitted.append((lane, len(roots) if interval['sensor_integrating'] else 0))
    ray_count = sum(count for _, count in admitted)
    barrier.record_emission_work_dispatched(slice_id, ray_count)
    for lane, count in admitted:
        if count:
            renderer._gpu_pump_prebuilt(lane['state'], n_rays=count)
    finish()
    barrier.confirm_transport_products_materialized(slice_id)
    submitted_sites = 0
    if interval['sensor_integrating']:
        if not barrier.sensor_may_submit(slice_id):
            raise RuntimeError('timed exposure transport was not materialized')
        weight = (float(interval['t1']) - float(interval['t0'])) / sweeps
        for lane in lanes:
            sensor = lane['accumulator']
            sensor.set_exposure_interval_weight(weight)
            target_pass = sensor._pass + sweeps
            while sensor._pass < target_pass:
                sensor.tick(strips=1)
            submitted_sites += sweeps * sensor._w * sensor._h
        barrier.record_sensor_probe_work_dispatched(slice_id, submitted_sites)
        finish()
    barrier.confirm_reception_closed(slice_id)
    if interval['sensor_integrating']:
        barrier.confirm_detector_integration_committed(slice_id)
    return ray_count, submitted_sites


def _camera_for_frame(camera, frame_index, total_frames, poses=None):
    if poses is not None:
        return dict(poses[frame_index])
    eye = np.asarray(camera["eye_m"], np.float32)
    target = np.asarray(camera["target_m"], np.float32)
    angle = 2.0 * np.pi * frame_index / total_frames
    dx, dy, dz = eye - target
    camera_eye = target + np.asarray((np.cos(angle) * dx + np.sin(angle) * dz,
                                      dy, -np.sin(angle) * dx + np.cos(angle) * dz), np.float32)
    return {**camera, "eye_m": camera_eye.tolist(), "target_m": target.tolist()}


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


def _sparse_sensor_regions(linear, spec):
    """Reduce declared chart pixels from the already-read linear snapshot."""
    if spec is None:
        return None
    values = np.asarray(linear)
    if values.ndim == 2:
        values = values[:, :, None]
    if (values.ndim != 3 or values.shape[:2] != (spec['height'], spec['width'])
            or values.shape[2] not in (1, 3)):
        raise ValueError('calibration map and linear sensor shape differ')
    flat = values.reshape(-1, values.shape[2])
    result = {}
    for name, runs in spec['regions'].items():
        samples = sum(length for _, length in runs)
        if samples < 3:
            result[name] = {'pixels': samples, 'mean': None, 'usable': False}
            continue
        sums = np.zeros(values.shape[2], np.float64)
        for start, length in runs:
            selection = flat[start:start + length]
            if not np.isfinite(selection).all():
                raise ValueError('calibration region contains nonfinite linear samples')
            sums += selection.sum(axis=0, dtype=np.float64)
        result[name] = {'pixels': samples, 'mean': (sums / samples).tolist(),
                        'usable': True}
    return {'version': 1, 'space': 'raw accumulated linear sensor',
            'epoch': None, 'regions': result}


def _snapshot(accumulator, output: Path, digest: str, epoch: int,
              elapsed: float, preparation: float, rays: int, sensor_sites: int,
              finished: bool = False, transport: str = "fixed:1", *,
              frame_index: int = 0, total_frames: int = 1,
              frame_elapsed: float = 0.0, completed_frames: tuple[dict, ...] = (),
              gamma: float = 1.0, film_response: dict | None = None,
              display_params: dict | None = None, publish_status: bool = True,
              calibration_regions: dict | None = None, monochrome: bool = False) -> dict:
    from OpenGL.GL import GL_TEXTURE_2D, GL_RGBA, GL_FLOAT, glBindTexture, glGetTexImage, glFinish

    # Applied after each branch's own exposure/filmic tonemap, on the final
    # [0,1] display value: gamma>1 lifts midtones (the preview looked stark
    # because this stage previously did not exist at all for "fixed:1", and
    # was fixed at the sRGB 1/2.4 power for "fixed:3"). Gamma=1 adds no
    # artistic gamma adjustment after the output profile's sRGB transfer.
    inv_gamma = 1.0 / max(1.0e-3, float(gamma))

    color_profile = None
    coverage = None
    glFinish()
    if transport in ("fixed:3", "fixed:4", "fixed:8"):
        # FilmStack's ordered R/G/B emulsion layers are separate linear
        # RGBA32F exposure textures. Their RGB components repeat the scalar
        # exposure; read one component from each, preserving relative energy.
        layers = []
        for texture in accumulator._textures[:(8 if transport == "fixed:8" else 4 if transport == "fixed:4" else 3)]:
            glBindTexture(GL_TEXTURE_2D, texture)
            raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
            layer = np.frombuffer(raw, dtype=np.float32).reshape(
                accumulator._h, accumulator._w, 4)
            layers.append(layer[:, :, 0].copy())
            if getattr(accumulator, "_history_samples", 0):
                # sensor_store writes alpha=1 for every sampled pixel,
                # including a genuinely black/zero-radiance contribution.
                # clear() restores alpha=0. Luminance is not a coverage test.
                sampled = np.isfinite(layer[:, :, 3]) & (layer[:, :, 3] > 0)
                coverage = sampled.copy() if coverage is None else coverage & sampled
        nir = layers[6] if transport == "fixed:8" else layers[3] if transport == "fixed:4" else None
        linear = np.stack(layers[:3], axis=2)
        # Preserve raw sensor values. Color characterization precedes all
        # preview tone/gamma operations and is never applied to the HDR twice.
        if transport == "fixed:8":
            spectral = np.stack(layers, axis=2)
            linear_output, color_profile = project_spectral_exposure(
                spectral, (display_params or {}).get("spectral_look", "compressed"))
        else:
            linear_output, color_profile = sensor_to_linear_srgb(linear, film_response)
        display, white = render_linear_srgb(linear_output, {"gamma": gamma, **(display_params or {})})
        luma = np.maximum(linear_output @ SRGB_LUMA, 0.0)
        mode = "RGB"
        if monochrome:
            display = np.uint8(np.clip(np.mean(display, axis=2), 0, 255))
            mode = "L"
        hdr_name = f"exposure-{epoch:06d}.npz" if finished else None
        if finished:
            temporary_hdr = output / (hdr_name + ".tmp")
            with temporary_hdr.open("wb") as stream:
                response = (normalize_film_response() if film_response is None
                            else normalize_film_response(film_response))
                np.savez_compressed(
                    stream, linear_rgb_exposure=linear,
                    **({"linear_nir850_exposure": nir} if nir is not None else {}),
                    **({"linear_spectral_exposure": spectral} if transport == "fixed:8" else {}),
                    **({"linear_mono_exposure": luma} if monochrome else {}),
                    # Legacy discrete transport-band labels, not the shifted
                    # Gaussian centers or monochromatic channel measurements.
                    wavelengths_nm=np.asarray([650,550,450,590,365,395,850,940] if transport == "fixed:8" else [650,550,450,850] if nir is not None else [650,550,450],np.float32),
                    film_response_version=np.asarray(response["version"], np.int32),
                    film_centers_nm=np.asarray(response["centers_nm"], np.float64),
                    film_sigmas_oct=np.asarray(response["sigmas_oct"], np.float64),
                    # Keep the same precision used by live retouch so saved
                    # retouch cannot change a rounding-boundary pixel.
                    linear_srgb_exposure=linear_output,
                    color_profile_json=np.asarray(json.dumps(color_profile, sort_keys=True)),
                    geometry_stats_json=np.asarray(json.dumps(getattr(accumulator, "_geometry_stats", None), sort_keys=True)),
                )
            os.replace(temporary_hdr, output / hdr_name)
        # Live retouch: re-tonemap the SAME in-memory linear array this
        # snapshot already holds -- no file is written or read back for
        # this. retouch.json (written by the server when sliders change) is
        # just a tiny parameter handoff; the actual image reprocessing
        # happens here, straight off the live GPU texture read above, every
        # snapshot, mid-exposure or finished, same as the baked preview.
        retouch_path = output / "retouch.json"
        if retouch_path.exists():
            try:
                retouch_params = json.loads(retouch_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                retouch_params = None
            if retouch_params:
                retouch_linear = (project_spectral_exposure(spectral, retouch_params.get("spectral_look", "compressed"))[0]
                                  if transport == "fixed:8" else linear_output)
                retouched_name = _render_retouch(retouch_linear, retouch_params, output, colorimetric=True)
            else:
                retouched_name = None
        else:
            retouched_name = None
    else:
        retouched_name = None
        glBindTexture(GL_TEXTURE_2D, accumulator.tex)
        raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(raw, dtype=np.float32).reshape(
            accumulator._h, accumulator._w, 4).copy()
        luma = image[:, :, :3].mean(axis=2)
        nonzero = luma[luma > 0]
        clip = float((display_params or {}).get('highlight_clip_percent', 0.0))
        iso = float((display_params or {}).get('exposure_iso', 100.0))
        white = max(float(np.percentile(nonzero, 98 if not display_params else 100-clip)) if len(nonzero) else 0.0, 1e-6)
        display = np.clip((luma / white) * (iso / 100.0), 0, 1) ** inv_gamma
        display = np.uint8(display * 255)
        mode = "L"
        hdr_name = f"exposure-{epoch:06d}.npz" if finished else None
        if finished:
            temporary_hdr = output / (hdr_name + ".tmp")
            with temporary_hdr.open("wb") as stream:
                np.savez_compressed(stream, linear_mono_exposure=luma.astype(np.float32, copy=False),
                    transport=np.asarray("fixed:1"),
                    geometry_stats_json=np.asarray(json.dumps(getattr(accumulator, "_geometry_stats", None), sort_keys=True)))
            os.replace(temporary_hdr, output / hdr_name)
    if coverage is not None:
        # Keep the shared display transform's RGB bytes exactly. Only live
        # views reveal the underlying UI where no sensor sample exists yet.
        display = np.dstack((display, coverage.astype(np.uint8) * 255))
        mode = "RGBA"
    filename = f"preview-{epoch:06d}.png"
    temporary = output / (filename + ".tmp")
    png_info = None
    if color_profile is not None:
        png_info = PngInfo()
        png_info.add(b"sRGB", b"\x00")
    Image.fromarray(display, mode=mode).save(temporary, format="PNG", pnginfo=png_info)
    os.replace(temporary, output / filename)
    marker_name = None
    if transport in ("fixed:4", "fixed:8"):
        marker_name = f"marker-{epoch:06d}.png"
        positive = nir[nir > 0]
        marker_white = max(float(np.percentile(positive, 99)) if len(positive) else 0.0, 1e-9)
        marker = np.uint8(np.clip(nir / marker_white, 0, 1) * 255)
        marker_tmp = output / (marker_name + ".tmp")
        Image.fromarray(marker, mode="L").save(marker_tmp, format="PNG")
        os.replace(marker_tmp, output / marker_name)
    frames = [*completed_frames]
    if finished:
        frame = {"index": frame_index, "image": filename, "hdr": hdr_name}
        if getattr(accumulator, "_frame_camera", None) is not None:
            frame["camera"] = accumulator._frame_camera
        frames.append(frame)
    # Delete this epoch's now-superseded intermediate preview/HDR files.
    # Only the just-written frame and any already-"finished" frames (kept
    # for orbit playback) survive -- the progressive-preview history served
    # no purpose once a newer snapshot exists, and previously kept every
    # single intermediate render forever (confirmed: one long exposure left
    # 1,677 PNGs behind, 3.3 GiB, never cleaned up).
    keep = {filename}
    if marker_name:
        keep.add(marker_name)
    if hdr_name:
        keep.add(hdr_name)
    if retouched_name:
        keep.add(retouched_name)
    for frame in frames:
        keep.add(frame["image"])
        if frame.get("hdr"):
            keep.add(frame["hdr"])
    if getattr(accumulator, "_history_samples", 0):
        # A live frame can be fetched just after the next status publication.
        # Retain three delivery frames, not an unbounded PNG archive; the real
        # radiance history remains in the GPU ring, never in these images.
        recent = sorted(output.glob("preview-*.png"), key=lambda p: p.stat().st_mtime_ns)
        keep.update(p.name for p in recent[-3:])
    for stale in output.glob("preview-*.png"):
        if stale.name not in keep:
            try:
                stale.unlink()
            except OSError:
                pass
    for stale in output.glob("marker-*.png"):
        if stale.name not in keep:
            try:
                stale.unlink()
            except OSError:
                pass
    for stale in output.glob("exposure-*.npz"):
        if stale.name not in keep:
            try:
                stale.unlink()
            except OSError:
                pass
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
        "display_gamma": gamma,
        "transport": ("fixed:8@365,395,450,550,590,650,850,940nm" if transport == "fixed:8" else
                      "fixed:4@450,550,650,850nm" if transport == "fixed:4" else
                      "fixed:3@450,550,650nm" if transport == "fixed:3"
                      else "fixed:1@587.5618nm"),
        "camera": "Spectral analytical thin lens",
    }
    if getattr(accumulator, "_has_rod_bundle", False):
        from camera_designer.studio_rod_bundle_gpu import read_bundle_failures
        failed=read_bundle_failures(accumulator._ssbo_regions,accumulator._n_regions)
        status['rod_bundle_diagnostic']={'unresolved_samples':failed,
            'policy':'twist-following angular partitions among occupied axial rods',
            'warning':('Unresolved cut intersections terminated samples; image is not validated.' if failed else '')}
    if getattr(accumulator, "_geometry_stats", None) is not None:
        status["geometry_stats"] = accumulator._geometry_stats
    if getattr(accumulator, "_bounce_diagnostic", None) is not None:
        status["bounce_diagnostic"] = accumulator._bounce_diagnostic
        status["effective_config"] = "effective_config.json"
    if hdr_name:
        status["hdr"] = hdr_name
        status["hdr_channels"] = ("raw eight-band linear sensor exposure plus separate artistic false-color linear RGB"
                                  if transport == "fixed:8" else
                                  "raw linear sensor exposure plus separate profiled linear sRGB; one display transform"
                                  if transport in ("fixed:3", "fixed:4", "fixed:8") else "raw linear monochrome sensor exposure")
    if marker_name:
        status["marker"] = marker_name
    if monochrome:
        status["sensor_output"] = "mono"
    if color_profile is not None:
        status["color_profile"] = color_profile
        status["display_transform"] = display_transform_metadata(display_params, effective_white=white)
        if display_params is not None:
            status["live_display"] = display_params
        if retouched_name:
            status["retouch_params"] = retouch_params
            status["retouch_display_transform"] = display_transform_metadata(retouch_params,
                display_white_reference(retouch_linear, retouch_params))
            if transport == "fixed:8":
                status["retouch_display_transform"]["spectral_look"] = retouch_params.get("spectral_look", "compressed")
        if color_profile.get("warning"):
            status["color_profile_warning"] = color_profile["warning"]
    if film_response is not None:
        status["film_response"] = normalize_film_response(film_response)
        status["effective_config"] = "effective_config.json"
    if retouched_name:
        status["retouched_image"] = retouched_name
    if calibration_regions is not None:
        sparse = _sparse_sensor_regions(luma if monochrome or transport == 'fixed:1' else linear,
                                        calibration_regions)
        sparse['epoch'] = epoch
        status['calibration_regions'] = sparse
    if coverage is not None:
        status.update(sampled_pixels=int(np.count_nonzero(coverage)),
                      coverage_fraction=float(np.mean(coverage)),
                      coverage_kind="sensor sample alpha; sampled black is opaque")
    if publish_status:
        _atomic_json(output / "status.json", status)
    return status


# A fixed, repeatedly-overwritten filename let a browser (or this process's
# own in-page Image cache) treat two different retouches as "the same URL"
# and skip redrawing -- confirmed: Apply/slider changes landed correctly on
# disk but never appeared on screen. Every retouch now gets its own filename
# instead, drawn from a counter reserved above the real per-epoch preview
# range so it can never collide with one; _snapshot()'s existing stale-file
# sweep (unknown names not in `keep` get deleted) prunes the previous one.
_retouch_seq = itertools.count()


def _retouch_filename() -> str:
    return f"preview-{900000 + (next(_retouch_seq) % 100000):06d}.png"


def _render_retouch(linear: np.ndarray, params: dict, output: Path, *, colorimetric: bool = False) -> str:
    """Re-tonemap the live in-memory linear radiance with live retouch
    parameters, straight off the same GPU texture read _snapshot() already
    did this epoch -- no HDR file is read back from disk for this. Writes
    a fresh, uniquely-named file every call (see _retouch_filename) and
    returns its name.
    """
    if colorimetric:
        display, _ = render_linear_srgb(linear, params)
        filename = _retouch_filename()
        temporary = output / (filename + ".tmp")
        png_info = PngInfo()
        png_info.add(b"sRGB", b"\x00")
        Image.fromarray(display, mode="RGB").save(temporary, format="PNG", pnginfo=png_info)
        os.replace(temporary, output / filename)
        return filename
    luma = np.sum(linear * [0.2126, 0.7152, 0.0722], axis=2)
    nonzero = luma[luma > 0]
    base_white = max(float(np.percentile(nonzero, 99)) if len(nonzero) else 0.0, 1e-6)
    exposure = float(params.get("exposure", 0.0))
    gamma = max(1.0e-3, float(params.get("gamma", 1.0)))
    red = float(params.get("red", 1.0))
    green = float(params.get("green", 1.0))
    blue = float(params.get("blue", 1.0))
    white_balance = float(params.get("whitebalance", 0.0))
    tint = float(params.get("tint", 0.0))
    intensity = max(0.0, float(params.get("intensity", 1.0)))
    exp_mul = 2.0 ** exposure
    wb = 2.0 ** white_balance
    tn = 2.0 ** tint
    gain = np.asarray([red * exp_mul * wb, green * exp_mul / tn, blue * exp_mul / wb], np.float64)
    k = 2.0 / base_white
    mapped = 1.0 - np.exp(-np.maximum(linear * gain, 0.0) * k)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped,
                    1.055 * np.power(np.maximum(mapped, 0.0), 1.0 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0) ** (1.0 / gamma)
    srgb = np.clip(srgb * intensity, 0.0, 1.0)
    display = np.uint8(srgb * 255)
    filename = _retouch_filename()
    temporary = output / (filename + ".tmp")
    Image.fromarray(display, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, output / filename)
    return filename


def _fixed8_studio_tick_frames(frames, seconds, substeps):
    """Give every eight-band shutter tick its own ordered field/sensor slice.

    A static source frame is a held state, not permission to reuse one field
    dispatch for every sensor sweep. At 512 pixels that reuse yielded a black
    eight-band exposure; separate tick slices preserve the native lockstep.
    """
    tick = float(seconds) / int(substeps)
    result = []
    for frame_index, frame in enumerate(frames):
        start, end = float(frame['t0']), float(frame['t1'])
        cuts = [start]
        cuts.extend(tick * index for index in range(1, int(substeps))
                    if start + 1e-9 < tick * index < end - 1e-9)
        cuts.append(end)
        result.extend({**frame, 't0': t0, 't1': t1, '_held_frame_key': frame_index}
                      for t0, t1 in zip(cuts, cuts[1:]))
    return result


def _timed_exposure_program(frames, fixtures, seconds, trigger_graph=None, trigger_timing=None):
    """Attach Shop cutoff and routed trigger states to one camera exposure."""
    from camera_software.exposure_timing import ExposureSlice, ExposureFrame, CameraTimeline, ExposureBarrier
    triggers = fixtures.get('trigger_emitters', [])
    pulse_duration=0.0
    if triggers:
        from camera_software.trigger_bus import make_camera_trigger_graph, resolve_camera_trigger_bus
        timing=trigger_timing or {'shop_cutoff_lead_s':.001,'flash_duration_s':.010}
        pre=float(timing['shop_cutoff_lead_s'])
        pulse_duration=float(timing['flash_duration_s'])
        pulse_end=pre+pulse_duration
        if not 0<pre<pulse_end<seconds:raise ValueError('camera trigger timing is outside shutter interval')
        bounds = sorted({0.0, pre, pulse_end, float(seconds),
                         *(float(value) for frame in frames for value in (frame['t0'],frame['t1']))})
        intervals = []
        for t0,t1 in zip(bounds,bounds[1:]):
            if t1 <= t0:continue
            original = next((frame for frame in frames if frame['t0'] <= t0 and frame['t1'] >= t1), None)
            if original is None:raise ValueError('timed exposure frame bundle has a gap')
            profiles = {'version':1,'records':[dict(record) for record in original['emitter_profiles']['records']]}
            if t0 < pulse_end:
                for record in profiles['records']:
                    if record['material'] in fixtures['shop_material_ids']:record['gain']=0.0
            intervals.append({**original,'t0':t0,'t1':t1,'emitter_profiles':profiles,
                              'sensor_integrating':t0>=pre,'flash_active':pre<=t0<pulse_end})
    else:
        intervals=[{**frame,'sensor_integrating':True,'flash_active':False} for frame in frames]
    slices=tuple(ExposureSlice(frame_id=0,slice_id=i,t0=float(item['t0']),t1=float(item['t1']),
        shutter_mode=0,shutter_open=1.0,shutter_center_u=.5,shutter_center_v=.5,
        shutter_softness=0.0,exposure_weight=float(item['t1'])-float(item['t0']),
        flash_weight=((float(item['t1'])-float(item['t0']))/pulse_duration) if item['flash_active'] else 0.0,
        sensor_weight=(float(item['t1'])-float(item['t0'])) if item['sensor_integrating'] else 0.0,
        flash_active=item['flash_active'],sensor_integrating=item['sensor_integrating'],
        scene_version_id=i,camera_id='studio-photo',causality_id=i)
        for i,item in enumerate(intervals))
    frame=ExposureFrame(frame_id=0,t0=0.0,t1=float(seconds),slices=slices,camera_id='studio-photo')
    if triggers:
        receivers=[item['receiver'] for item in triggers]
        if trigger_graph is None:
            graph=make_camera_trigger_graph(receivers)
            for receiver in receivers:graph.set_weight('Flash',receiver,1.0)
        else:
            from routing_engine import RoutingGraph
            graph=RoutingGraph.from_dict(trigger_graph)
        routed=resolve_camera_trigger_bus(frame,graph,receivers)
        for index,item in enumerate(intervals):
            if not item['flash_active']:continue
            by_material={record['material']:record for record in item['emitter_profiles']['records']}
            for trigger in triggers:
                if abs(routed.signals[trigger['receiver']][index])>1e-12:
                    if trigger['material'] not in by_material:
                        raise ValueError('triggered Shop material has no named emitter source')
                    by_material[trigger['material']]['gain']=trigger['gain']
    return intervals,ExposureBarrier(CameraTimeline.from_exposure_frame(frame))


def run(payload: Path, config_path: Path, output: Path, *,
        seconds: float = 600.0, snapshot_seconds: float = 2.0,
        max_epochs: int | None = None, avi_writer_root: Path | None = None,
        dynamic_scene_receiver=None) -> dict:
    if not 0 < seconds <= 3600:
        raise ValueError("photo exposure per frame must be in (0, 3600] seconds")
    if not 0.25 <= snapshot_seconds <= 60:
        raise ValueError("snapshot interval must be in [0.25, 60] seconds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    from camera_designer.marble_tiles import tile_plan, tile_windows, require_tile_disk, TileAssembly
    camera_poses = _camera_pose_sequence(config)
    lane_cameras = _sensor_lane_cameras(config)
    tiles = tile_plan(config)
    if tiles:
        output.mkdir(parents=True, exist_ok=True)
        require_tile_disk(output, tiles)
    cooperative=config.get('client_intersections',False)
    if type(cooperative) is not bool:raise ValueError('client_intersections must be boolean')
    if cooperative and (config.get('transport')!='fixed:3' or config.get('orbit_frames',1)!=1):
        raise ValueError('Cooperative intersections currently require fixed:3 transport with one camera frame')
    max_bounces = _bounce_limit(config.get("max_bounces", 4))
    # Older direct worker jobs predate Auto; new Studio requests send the mode.
    bounce_mode = config.get("bounce_mode", "manual")
    if bounce_mode not in ("auto", "manual"):
        raise ValueError("bounce_mode must be auto or manual")
    orbit_frames = config.get("orbit_frames", 1)
    if type(orbit_frames) is not int or orbit_frames not in (1, 8, 16, 32):
        raise ValueError("orbit_frames must be 1, 8, 16, or 32")
    total_frames = len(camera_poses) if camera_poses is not None else orbit_frames
    transport = str(config.get("transport", "fixed:1"))
    from camera_designer.studio_forward_field import ForwardFieldPublisher, validate_receive_light_field
    receive_light_field = validate_receive_light_field(config)
    response_requested=config.get("light_field_information")=="evaluated-responses-v1"
    if response_requested:
        raise RuntimeError("evaluated scene responses execution held: hardware compile resource validation pending")
    if response_requested and (not receive_light_field or transport!="fixed:3" or tiles or cooperative or bounce_mode!="manual"):
        raise ValueError("evaluated scene responses require explicit fixed:3 field, untiled native sensor and fixed bounce budget")
    from camera_designer.studio_opaque_pbr import validate_surface_transport
    surface_transport=validate_surface_transport(config)
    if transport not in ("fixed:1", "fixed:3", "fixed:4", "fixed:8"):
        raise ValueError("transport must be fixed:1, fixed:3, fixed:4 or fixed:8")
    if transport == "fixed:1" and "film_response" in config:
        raise ValueError("film_response requires fixed:3 RGB transport")
    film_response = None
    if transport in ("fixed:3", "fixed:4", "fixed:8"):
        film_response = (normalize_film_response(config["film_response"])
                         if "film_response" in config else normalize_film_response())
    gamma = float(config.get("gamma", 1.0))
    if not 0.2 <= gamma <= 5.0:
        raise ValueError("gamma must be in [0.2, 5.0]")
    exposure_substeps=config.get('exposure_substeps',int(np.ceil(seconds*60)))
    exposure_iso=float(config.get('exposure_iso',100.0))
    highlight_clip_percent=float(config.get('highlight_clip_percent',25.0))
    if (type(exposure_substeps) is not int or not int(np.ceil(seconds*10-1e-9))<=exposure_substeps<=216000
            or not np.isfinite(exposure_iso) or not 25<=exposure_iso<=6400
            or not np.isfinite(highlight_clip_percent) or not 0<=highlight_clip_percent<=25):
        raise ValueError('invalid exposure substeps, ISO, or highlight clip')
    display_params=({'normalization':'percentile','normalization_percentile':100-highlight_clip_percent,
                     'tone_curve':'linear','exposure':float(np.log2(exposure_iso/100)),
                     'exposure_iso':exposure_iso,'highlight_clip_percent':highlight_clip_percent}
                    if 'exposure_iso' in config or 'highlight_clip_percent' in config else None)
    wavelengths = ([.5875618] if transport == "fixed:1" else
                   [.650, .550, .450, .590, .365, .395, .850, .940] if transport == "fixed:8" else
                   [.650, .550, .450, .850] if transport == "fixed:4" else [.450, .550, .650])
    digest = hashlib.sha256(payload.read_bytes() +
                            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    field_publisher = ForwardFieldPublisher(output, digest) if receive_light_field else None
    # Preserve the caller's digest/config identity, while recording defaults
    # explicitly for reproducibility of old requests that omit this field.
    effective_config = dict(config)
    if film_response is not None:
        effective_config["film_response"] = film_response
    _atomic_json(output / "effective_config.json", effective_config)
    preparation_status = {
        "digest": digest, "state": "preparing", "image": None,
        "frames": [], "frame_index": 0, "total_frames": total_frames,
        **({"tiled": tiles} if tiles else {}),
        "transport": ("fixed:8@365,395,450,550,590,650,850,940nm" if transport == "fixed:8" else
                      "fixed:4@450,550,650,850nm" if transport == "fixed:4" else
                      "fixed:3@450,550,650nm" if transport == "fixed:3"
                      else "fixed:1@587.5618nm"),
        **({"film_response": film_response, "effective_config": "effective_config.json"}
           if film_response is not None else {}),
    }
    _atomic_json(output / "status.json", preparation_status)
    import demo_pluck_gl as renderer
    with np.load(payload, allow_pickle=False) as saved:
        vertices = np.asarray(saved["vertices"], np.float32)
        material_ids = np.asarray(saved["material_ids"], np.int32)
        pbr = np.asarray(saved["pbr"], np.float32)
        from camera_designer.studio_volume_extinction import from_payload
        volume_extinction = from_payload(config, saved)
    if volume_extinction is not None and config.get('light_field_internal_media',False):
        raise ValueError('internal directional recording with ExtinctionGrid requires joint attenuation integration')
    if volume_extinction is not None:
        effective_config['volume_extinction_field'] = volume_extinction.metadata()

    preparation_start = time.monotonic()
    from camera_designer.studio_photo_geometry import (
        CAMERA_GEOMETRY_VERSION, camera_geometry_key, prepare_photo_geometry,
    )
    initial_config = {**config, "camera": camera_poses[0]} if camera_poses is not None else config
    timed_frames = config.get("exposure_state_frames")
    timed_barrier = None
    if lane_cameras is not None and timed_frames is None:
        raise ValueError("simultaneous sensor lanes require a timed exposure")
    if timed_frames is not None:
        if (not isinstance(timed_frames, list) or not timed_frames or transport not in ("fixed:3", "fixed:4", "fixed:8")
                or config.get("live") or tiles or total_frames != 1 or cooperative or receive_light_field):
            raise ValueError("timed exposure requires one ordinary RGB fixture photograph")
        import copy
        initial_camera=initial_config['camera']
        initial_config={**initial_config,'camera':{**initial_camera,
            'eye_m':np.asarray(initial_camera.get('eye_m',[0,0,.07]),np.float32).tolist(),
            'target_m':np.asarray(initial_camera.get('target_m',config.get('checker',{}).get('center_m',[0,0,.30])),np.float32).tolist()}}
        initial_config = {**initial_config, 'fixtures': copy.deepcopy(initial_config['fixtures'])}
        fixture = initial_config["fixtures"]
        if not fixture.get("emissive_material_ids"):
            raise ValueError("timed exposure needs its canonical emitter material identities")
        for index,material in enumerate(fixture["emissive_material_ids"]):
            value = 1.0 + (index + 1) * .01
            fixture["pbr"][material * 16 + 8:material * 16 + 11] = [value, value, value]
        for index,record in enumerate(fixture.get("emitter_profiles", {}).get("records", [])):
            record["gain"] = 1.0 + (index + 1) * .01
            record["filter"] = [1.0, 1.0, 1.0]
        if abs(timed_frames[0]['t0'])>1e-5 or abs(timed_frames[-1]['t1']-seconds)>1e-5 or any(
                item['t1']<=item['t0'] or (i and abs(item['t0']-timed_frames[i-1]['t1'])>1e-5)
                for i,item in enumerate(timed_frames)):
            raise ValueError('timed exposure slices must cover the shutter interval')
        if transport == 'fixed:8' and exposure_substeps > 1:
            timed_frames = _fixed8_studio_tick_frames(timed_frames, seconds, exposure_substeps)
        timed_frames,timed_barrier=_timed_exposure_program(timed_frames,config['fixtures'],seconds,
            config.get('camera_trigger_graph'),config.get('camera_trigger_timing'))
    material_lineage = [] if receive_light_field else None
    response_lineage={} if response_requested else None
    scene, sphere_regions, geometry_stats, checker_pos = prepare_photo_geometry(initial_config, vertices, material_ids, pbr, **({"material_lineage_out": material_lineage} if material_lineage is not None else {}), **({"response_lineage_out":response_lineage} if response_requested else {}))
    geom, shade, mat_buf, triangles, contexts, _, bounds, sources = scene
    if surface_transport is not None:geometry_stats['surface_transport']=surface_transport
    preparation_status["geometry_stats"] = geometry_stats
    effective_config["geometry_stats"] = geometry_stats
    effective_config["camera_geometry_version"] = CAMERA_GEOMETRY_VERSION
    _atomic_json(output / "effective_config.json", effective_config)
    _atomic_json(output / "status.json", preparation_status)
    recording_region = None
    if config.get('light_field_region') is not None:
        if config['light_field_region']!='subject':raise ValueError('only bounded subject recording ROI is supported')
        from camera_designer.studio_directional_field import subject_field_region
        recording_region=subject_field_region(vertices,config)
        if config.get('light_field_subject_region') not in (None,recording_region):raise ValueError('subject ROI receipt differs from actual geometry')
    renderer._ensure_worker_gl_context()
    margin = np.asarray([.02, .02, .04], np.float32)
    bounds = (bounds[0] - margin, bounds[1] + margin)
    nodes, ids = renderer._build_gpu_bvh(triangles)
    # Keep the existing fixed:3 preset unless the job supplies sensitivity
    # controls. This is a custom sensor, not a standard color profile. Copy
    # its layers so changing one job cannot mutate the shared preset. Both
    # paths receive this SAME stack: its fixed transport basis isolates the
    # material optics from the response applied once at the final sensor.
    film = (build_film_stack(renderer.FILM_STACKS["em_rgb_fixed3"], film_response)
            if film_response is not None else None)
    if transport in ("fixed:4", "fixed:8"):
        extras = ([("amber590",590.),("UVA365",365.),("UVA395",395.),
                   ("NIR850",850.),("NIR940",940.)] if transport == "fixed:8" else [("NIR850",850.)])
        for name,wavelength_nm in extras:
            film.add(renderer.FilmLayer(name, "em",
                response_center_hz=299792458.0 / (wavelength_nm*1e-9),
                response_width_oct=.04))
        film.transport_wavelengths_nm = tuple(w*1000 for w in wavelengths)
    # Bounce budget: exposed in the Studio UI ("Light bounces") because
    # glass/gem subjects and the new reflective room both need real,
    # user-controlled transport depth, not one silently fixed number.
    # Validated before creating the GL context. Both existing dynamic shader
    # loops receive the full requested budget, including 200. This does not
    # force paths to survive: their existing termination rules still apply.
    from camera_designer.studio_region_gpu import region_records_from_geometry
    from camera_designer.studio_rod_bundle_gpu import pack_bundle_records
    region_records=region_records_from_geometry(geom)
    authored_bundle=config.get('region_media',{}).get('axial_bundle')
    if authored_bundle is not None:
        region_records=pack_bundle_records(region_records,authored_bundle,
            scale_m=config.get('scale_m',.025),center_m=config.get('object_center_m',[0,0,.18]),
            host_triangle=int(np.flatnonzero(geom[:,3]==-1)[0]))
    response_consumer=None
    response_publisher=None
    response_holder={'state':None}
    response_reservation=None
    if response_requested:
        from camera_designer.studio_response_factory import SceneResponseConsumer,selected_sensor_reservation,retained_scene_identity
        from camera_designer.studio_scene_response_publish import SceneResponsePublisher
        retention_bytes=16*1024*1024
        application_bytes=8*1024*1024
        response_reservation=selected_sensor_reservation(config,sources,retention_bytes=retention_bytes+application_bytes)
        indexing_bounds=([recording_region['bounds_min_m'],recording_region['bounds_max_m']] if recording_region else bounds)
        indexing_dims=recording_region['dimensions'] if recording_region else [config.get('light_field_grid_size',64)]*3
        def current_response_lineage():
            current=response_holder['state']
            return current['response_lineage'] if current is not None else response_lineage
        response_consumer=SceneResponseConsumer(memory_budget_bytes=retention_bytes,lineage_provider=current_response_lineage,
            bounds=indexing_bounds,dimensions=indexing_dims,stencil_size=config.get('light_field_stencil',12))
        response_publisher=SceneResponsePublisher(output,digest)
    from material_db import MaterialDatabase
    state = renderer._gpu_ray_field_prebuilt(
        geom, shade, mat_buf, nodes, ids, contexts, sources, bounds,
        dims=(96, 96, 96), dispatch_batch=512, max_bounces=max_bounces, initial_rays=128,
        film_stack=film, sphere_regions=sphere_regions, region_records=region_records,
        native_pbr=MaterialDatabase.instance().build_tensors()["pbr"],
        record_directional=validate_receive_light_field(config),
        light_field_stencil=config.get("light_field_stencil",12),
        light_field_estimator=config.get("light_field_estimator","track-length-throughput-v1"),
        light_field_grid_size=config.get("light_field_grid_size"),
        light_field_region=recording_region,
        surface_transport=surface_transport,
        light_field_internal_media=config.get("light_field_internal_media",False),
        light_field_information=config.get("light_field_information"),
        **(dict(forward_response_consumer=response_consumer,forward_response_identity=dict(scene_revision=retained_scene_identity(config,vertices,material_ids,pbr),population_index=0,source_population_seed=config.get('emitter_sampling',{}).get('seed',42)),
            forward_response_reserved_bytes=response_reservation['total']) if response_requested else {}),
        directional_quantization_divisor=(16777216 if config.get('emitter_sampling',{}).get('refresh_each_sweep') else 65535),
        volume_extinction=volume_extinction,
    )
    if state is None:
        raise RuntimeError("Spectral GPU ray-field shader did not link")
    if response_requested:
        state["response_lineage"]=response_lineage
        response_holder['state']=state
        state['scene_response_consumer']=response_consumer
        state['scene_response_publisher']=response_publisher
        state['scene_response_reservation']=response_reservation
        def build_recorded_application(store):
            from camera_designer.studio_scene_application import build_scene_application
            from camera_designer.studio_response_factory import current_scene_query_metadata
            current=current_scene_query_metadata(state,response_consumer,state.get('response_live_revision',0))
            try:return build_scene_application(store,film_response=film_response,compile_program=renderer._prog,memory_budget_bytes=application_bytes,current_metadata=current)
            finally:renderer._restore_prebuilt_bindings(state)
        response_consumer.application_builder=build_recorded_application
        state["_response_extra_arrays"]=(scene,nodes,ids,vertices,material_ids,pbr)
    if receive_light_field:
        state["geometry_sha256"] = config.get("geometry_sha256")
        state["directional_record"]["material_lineage"] = material_lineage
    bounce_diagnostic = {"mode": "manual", "chosen_depth": max_bounces, "render_cap": 200}
    if bounce_mode == "auto":
        from camera_designer.marble_bounce_probe import run_bounce_probe
        def probe_progress(batch, batches):
            _atomic_json(output / "status.json", {
                **preparation_status,
                "bounce_diagnostic": {"mode": "auto", "state": "probing", "batch": batch,
                                      "batches": batches, "probe_depth": 1000, "render_cap": 200},
            })
        try:
            bounce_diagnostic = run_bounce_probe(renderer, state, bounds, wavelengths,
                n_triangles=len(triangles), n_nodes=len(nodes), context_count=len(contexts),
                progress=probe_progress)
        except (ValueError, RuntimeError) as error:
            bounce_diagnostic = {"mode": "auto", "chosen_depth": 200, "render_cap": 200,
                "probe_depth": 1000, "confidence": "inconclusive pilot",
                "warning": f"Auto probe unavailable: {error}. Render cap 200 retained; convergence is unknown."}
        max_bounces = bounce_diagnostic["chosen_depth"]
        from OpenGL.GL import glUseProgram, glUniform1i, glGetUniformLocation
        glUseProgram(state["prog"])
        glUniform1i(glGetUniformLocation(state["prog"], b"uMaxBounces"), max_bounces)
        glUseProgram(0)
        # Initial field warmup used the request's old/manual count. Do not
        # mix it with the chosen global depth or with any diagnostic rays.
        renderer._clear_prebuilt(state)
    effective_config.update(bounce_mode=bounce_mode, max_bounces=max_bounces,
                            requested_manual_bounces=config.get("max_bounces", 4),
                            bounce_diagnostic=bounce_diagnostic)
    field_ray_jobs = 0 if bounce_mode == 'auto' or not state.get('n_sources') else 128
    _atomic_json(output / "effective_config.json", effective_config)
    _atomic_json(output / "status.json", {**preparation_status,
        "bounce_diagnostic": bounce_diagnostic, "effective_config": "effective_config.json"})
    print(f"  [marble_bounces] {bounce_mode}: render={max_bounces}/200; "
          f"probe={bounce_diagnostic.get('probe_depth', 'not requested')}; "
          f"{bounce_diagnostic.get('warning', '')}", flush=True)
    camera = initial_config.get("camera", {})
    resolution = int(camera.get("resolution", 256))
    if not tiles and resolution not in (128, 256, 512, 768, 1024, 2048, 4096):
        raise ValueError("preview resolution must be 128, 256, 512, 768, 1024, 2048, or 4096")
    live = config.get("live")
    width = int(live["width"]) if live else tiles["width"] if tiles else resolution
    height = int(live["height"]) if live else tiles["height"] if tiles else resolution
    if live:
        from camera_designer.sensor_history import SensorHistoryLayout
        if transport != "fixed:3" or total_frames != 1 or width > 512 or height > 512:
            raise ValueError("live view requires a bounded RGB single-camera sensor")
        SensorHistoryLayout(width, height, 3, live["history_samples"], live.get("checkpoint", False))
    eye = np.asarray(camera.get("eye_m", [0, 0, .07]), np.float32)
    target = np.asarray(camera.get("target_m", checker_pos), np.float32)
    camera = {**camera, "eye_m": eye.tolist(), "target_m": target.tolist()}
    active_camera_key = camera_geometry_key(camera, checker_pos)

    def update_camera_geometry(next_camera, active_sensor=None):
        nonlocal active_camera_key, geometry_stats, triangles, nodes, sphere_regions, scene
        nonlocal field_ray_jobs
        next_key = camera_geometry_key(next_camera, checker_pos)
        if next_key == active_camera_key:
            return
        next_config = {**config, "camera": next_camera}
        next_lineage = [] if receive_light_field else None
        next_response_lineage={} if response_requested else None
        next_scene, next_spheres, next_stats, _ = prepare_photo_geometry(
            next_config, vertices, material_ids, pbr, **({"material_lineage_out": next_lineage} if next_lineage is not None else {}), **({"response_lineage_out":next_response_lineage} if response_requested else {}))
        next_geom, next_shade, next_mat, next_tris, next_ctx, _, next_bounds, _ = next_scene
        next_bounds = (next_bounds[0] - margin, next_bounds[1] + margin)
        next_nodes, next_ids = renderer._build_gpu_bvh(next_tris)
        renderer._update_prebuilt_geometry(state, next_geom, next_shade, next_mat,
            next_nodes, next_ids, next_ctx, next_bounds, next_spheres,
            native_pbr=MaterialDatabase.instance().build_tensors()["pbr"])
        if receive_light_field:
            state["directional_record"]["material_lineage"] = next_lineage
        if response_requested:state["response_lineage"]=next_response_lineage
        if active_sensor is not None:
            active_sensor.set_scene_geometry(n_tris=len(next_tris), n_bvh_nodes=len(next_nodes),
                bmin=next_bounds[0], bmax=next_bounds[1], n_spheres=len(next_spheres))
            active_sensor._geometry_stats = next_stats
        scene = next_scene
        geometry_stats, triangles, nodes, sphere_regions = next_stats, next_tris, next_nodes, next_spheres
        if response_requested:state["_response_extra_arrays"]=(scene,nodes,next_ids,vertices,material_ids,pbr)
        active_camera_key = next_key
        # Old-pose photons cannot survive into an orbit frame or live sweep.
        # Preserve stochastic sequence, but never mix two different scenes.
        seed = state["_pump_seed"]
        renderer._clear_prebuilt(state)
        state["_pump_seed"] = seed
        field_ray_jobs = 0
        if field_publisher is not None:
            field_publisher.invalidate()

    def sensor(frame_index: int, window=None, *, lane=None):
        sensor_width = window["width"] if window else width
        sensor_height = window["height"] if window else height
        frame_camera = (lane['camera'] if lane is not None else
                        _camera_for_frame(camera, frame_index, total_frames, camera_poses))
        camera_eye = np.asarray(frame_camera["eye_m"], np.float32)
        camera_target = np.asarray(frame_camera["target_m"], np.float32)
        if lane is None:
            update_camera_geometry(frame_camera)
        sensor_state = lane['state'] if lane is not None else state
        sensor_transport = lane['transport'] if lane is not None else transport
        sensor_film = film if sensor_transport in ('fixed:3','fixed:4','fixed:8') else None
        sensor_scene = lane['scene'] if lane is not None else scene
        sensor_nodes = lane['nodes'] if lane is not None else nodes
        sensor_spheres = lane['spheres'] if lane is not None else sphere_regions
        sensor_stats = lane['stats'] if lane is not None else geometry_stats
        device_limits=_sensor_device_limits(sensor_width,sensor_height)
        print(json.dumps({"sensor_device_limits":device_limits}),flush=True)
        response_owner=None
        if response_requested:
            from camera_designer.studio_response_factory import make_response_owner
            response_owner=make_response_owner(config,sensor_state,frame_camera,consume=response_consumer.consume_camera_samples,
                extra_arrays=sensor_state['_response_extra_arrays'],completed_samples=True,
                sensor_reserved_bytes=response_reservation['total']-response_consumer.memory_budget_bytes-application_bytes,
                retention_reserved_bytes=response_consumer.memory_budget_bytes+application_bytes)
            response_owner._response_extra_arrays=sensor_state['_response_extra_arrays']
            response_owner._retention_reserved_bytes=response_consumer.memory_budget_bytes+application_bytes
            response_owner.identity['film_response']=film_response
            sensor_state['response_camera_owner']=response_owner
        accumulator = renderer.SensorAccumulator(
            ssbo_tris=sensor_state["ssbo"][0], ssbo_nodes=sensor_state["ssbo"][1],
            ssbo_ids=sensor_state["ssbo"][2], ssbo_shade=sensor_state["ssbo"][9],
            ssbo_mat=sensor_state["ssbo"][10], n_tris=len(sensor_scene[3]),
            n_bvh_nodes=len(sensor_nodes), tex_bands=sensor_state["tex_bands"],
            bmin=sensor_state["bmin"], bmax=sensor_state["bmax"], dims=sensor_state["dims"],
            camera_eye=camera_eye, camera_target=camera_target,
            sensor_w=sensor_width, sensor_h=sensor_height,
            **({"full_sensor_size": (width, height), "sensor_origin": (window["x"], window["y"])} if window else {}),
            rows_per_frame=(max(1,4096//width) if cooperative and not live else 16),
            fov_deg=float(frame_camera.get("fov_degrees", 20)), max_bounces=max_bounces,
            aperture_radius=float(frame_camera.get("aperture_radius_m", .005)),
            focus_dist=float(frame_camera.get("focus_distance_m", .23)),
            n_blades=int(frame_camera.get("aperture_blades", 0)),
            aperture_rot=float(frame_camera.get("aperture_rotation_radians", 0)),
            vol_alpha=0.0, vol_steps=8,
            n_sources=len(sensor_scene[7]),
            ssbo_spheres=sensor_state["ssbo"][12], n_spheres=len(sensor_spheres),
            ssbo_regions=sensor_state["ssbo"][15] if sensor_state.get("region_count") else 0,
            n_regions=sensor_state.get("region_count",0),
            ssbo_native_pbr=sensor_state["native_pbr_ssbo"],
            volume_extinction=sensor_state.get("volume_extinction"),
            surface_transport=surface_transport,
            **({"light_field_information":response_owner} if response_owner is not None else {}),
        )
        if response_requested:sensor_state['response_camera_accumulator']=accumulator
        if not accumulator._active:
            accumulator.destroy()
            raise RuntimeError("Spectral sensor compute shader did not link")
        if sensor_film is not None:
            accumulator.film = sensor_film
        if sensor_transport == "fixed:3" or sensor_stats.get('emissive_sampling'):
            try:
                accumulator.set_nee_spectral_sources(sensor_scene[7], sensor_state["ssbo"][5],
                    area_scene=sensor_scene if sensor_stats.get('emissive_sampling') else None)
            except Exception:
                accumulator.destroy()
                raise
        accumulator._frame_camera = frame_camera
        accumulator._has_rod_bundle = authored_bundle is not None
        accumulator._geometry_stats = sensor_stats
        accumulator._bounce_diagnostic = bounce_diagnostic
        accumulator.half_life = 0.0
        accumulator._device_limits=device_limits
        accumulator._sprinkle = False  # existing deterministic row-priority mode
        accumulator._sensor_lane_index = 0 if lane is None else lane['index']
        accumulator._sensor_lane_count = len(lane_records) if lane is not None else 1
        if response_requested:
            from camera_designer.studio_response_factory import rebind_response_owner
            rebind_response_owner(accumulator,sensor_state,frame_camera)
        return accumulator
    preparation = time.monotonic() - preparation_start
    epoch = 0
    completed_frames: list[dict] = []
    accumulator = None
    lane_records = []
    assembly = None
    cooperative_pump=None
    if cooperative:
        from camera_designer.studio_cooperative_sensor import CooperativeSensorPump
        def cooperative_status(work_state,profile):
            path=output/'status.json'
            current=json.loads(path.read_text()) if path.exists() else {'digest':digest}
            current.update(state='exposing',client_work_state=work_state,client_work_profile=profile)
            _atomic_json(path,current)
        cooperative_pump=CooperativeSensorPump(renderer,digest,on_state=cooperative_status)
    try:
        from OpenGL.GL import glFinish
        job_start = time.monotonic()
        if lane_cameras is not None:
            from camera_designer.studio_native_emitters import sampling_settings
            sampling = sampling_settings(config)
            lane_records.append(dict(index=0, camera=lane_cameras[0], scene=scene,
                spheres=sphere_regions, stats=geometry_stats, nodes=nodes, state=state,
                roots=scene[7], transport=transport, accumulator=None))
            for lane_index, lane_camera in enumerate(lane_cameras[1:], 1):
                lane_sampling = {**sampling, 'seed': (sampling['seed'] + lane_index) % 2147483648}
                lane_config = {**initial_config, 'camera': lane_camera,
                    'emitter_sampling': lane_sampling,
                    'transport': lane_camera.get('sensor_transport', transport)}
                lane_scene, lane_spheres, lane_stats, _ = prepare_photo_geometry(
                    lane_config, vertices, material_ids, pbr)
                lane_geom, lane_shade, lane_mat, lane_tris, lane_contexts, _, lane_bounds, lane_sources = lane_scene
                lane_bounds = (lane_bounds[0] - margin, lane_bounds[1] + margin)
                lane_nodes, lane_ids = renderer._build_gpu_bvh(lane_tris)
                lane_regions = region_records_from_geometry(lane_geom)
                if authored_bundle is not None:
                    lane_regions = pack_bundle_records(lane_regions, authored_bundle,
                        scale_m=config.get('scale_m', .025),
                        center_m=config.get('object_center_m', [0, 0, .18]),
                        host_triangle=int(np.flatnonzero(lane_geom[:, 3] == -1)[0]))
                lane_state = renderer._gpu_ray_field_prebuilt(
                    lane_geom, lane_shade, lane_mat, lane_nodes, lane_ids,
                    lane_contexts, lane_sources, lane_bounds,
                    dims=(96, 96, 96), dispatch_batch=512, max_bounces=max_bounces,
                    initial_rays=128,
                    film_stack=film if lane_config['transport'] in ('fixed:3','fixed:4','fixed:8') else None,
                    sphere_regions=lane_spheres,
                    region_records=lane_regions,
                    native_pbr=MaterialDatabase.instance().build_tensors()['pbr'],
                    surface_transport=surface_transport, volume_extinction=volume_extinction)
                if lane_state is None:
                    raise RuntimeError(f"sensor lane {lane_index} forward shader did not link")
                lane_records.append(dict(index=lane_index, camera=lane_camera,
                    scene=lane_scene, spheres=lane_spheres, stats=lane_stats,
                    nodes=lane_nodes, state=lane_state, roots=lane_sources,
                    transport=lane_config['transport'], accumulator=None))
            _require_disjoint_lane_roots([lane['scene'] for lane in lane_records])
        assembly = TileAssembly(output, tiles) if tiles else None
        tiled_sensor_sites = 0
        if live:
            from camera_designer.marble_live_worker import run_live_sensor
            accumulator = sensor(0)
            dynamic_reader = None
            if dynamic_scene_receiver is not None:
                # Bind to this worker's actual loaded config, material rows,
                # geometry and source buffer. Never adopt a claimed previous
                # Studio capture as proof that an Exposure worker matches it.
                from camera_designer.studio_dynamic_reader import StudioDynamicReader
                dynamic_reader = StudioDynamicReader(renderer, state, config, scene, material_ids, pbr)
                dynamic_scene_receiver(dynamic_reader)
            from camera_designer.studio_native_emitters import sampling_settings,install_source_population
            source_settings=sampling_settings(config)
            refresh_sources=None
            if source_settings['refresh_each_sweep']:
                if dynamic_reader is not None:raise ValueError('fresh source populations with dynamic subject updates require a new candidate')
                def refresh_sources(index,ray_jobs):
                    return install_source_population(renderer,state,accumulator,scene,wavelengths,source_settings,index,ray_jobs=ray_jobs)
            return run_live_sensor(accumulator, renderer, state, output, digest,
                                   config, preparation, _snapshot, _atomic_json,
                                   max_epochs=max_epochs,
                                   update_camera_geometry=update_camera_geometry,
                                   dynamic_scene_reader=dynamic_reader,
                                   refresh_sources=refresh_sources)
        windows = list(tile_windows(tiles)) if tiles else [None] * total_frames
        for work_index, window in enumerate(windows):
            frame_index = 0 if tiles else work_index
            work_output = output
            if tiles:
                work_output = output / f"tile-{work_index:06d}"
                work_output.mkdir(exist_ok=True)
                tiles = {**tiles, "active_tile": {"index": work_index, **window}}
                _atomic_json(output / "status.json", {**preparation_status, "state": "exposing", "tiled": tiles})
            accumulator = sensor(frame_index, window,
                lane=lane_records[0] if lane_records else None)
            if timed_frames is not None:
                from OpenGL.GL import glFinish
                if not lane_records:
                    lane_records.append(dict(index=0, camera=camera, scene=scene,
                        spheres=sphere_regions, stats=geometry_stats, nodes=nodes,
                        state=state, roots=scene[7], transport=transport, accumulator=None))
                lane_records[0]['accumulator'] = accumulator
                for lane in lane_records[1:]:
                    lane['accumulator'] = sensor(frame_index, window, lane=lane)
                    (output / f"lane-{lane['index']:02d}").mkdir(exist_ok=True)
                frame_start = time.monotonic()
                last_snapshot = frame_start
                calibration_history = []
                field_ray_jobs = 0
                sensor_sites = 0
                prepared_scene_cache = {'key': None, 'scenes': {}}
                active_fixture_geometry={'vertices':config['fixtures']['vertices'],
                                         'material_ids':config['fixtures']['material_ids']}
                geometry_revision=0
                last_geometry_capture=None
                for lane in lane_records:
                    lane['installed_fixture_geometry']=active_fixture_geometry
                    lane['installed_geometry_revision']=0
                    lane['installed_spheres']=lane['spheres']
                for interval_index, interval in enumerate(timed_frames):
                    dt = float(interval['t1']) - float(interval['t0'])
                    if not np.isfinite(dt) or dt <= 0:
                        raise ValueError('timed exposure needs positive finite state intervals')
                    held_key=interval.get('_held_frame_key',interval_index)
                    if 'vertices' in interval and held_key!=last_geometry_capture:
                        active_fixture_geometry={'vertices':interval['vertices'],
                                                 'material_ids':interval['material_ids']}
                        geometry_revision+=1
                        last_geometry_capture=held_key
                    fixture = {**config['fixtures'], **active_fixture_geometry, 'pbr': interval['pbr'],
                               'emitter_profiles': interval['emitter_profiles']}
                    def prepare_lane_scene(lane_camera, fallback_geometry=None):
                        held_key = interval.get('_held_frame_key', interval_index)
                        if prepared_scene_cache['key'] != held_key:
                            prepared_scene_cache['key'] = held_key
                            prepared_scene_cache['scenes'].clear()
                        lane_key = (id(lane_camera),id(fallback_geometry))
                        if lane_key not in prepared_scene_cache['scenes']:
                            lane_fixture=({**fixture,**fallback_geometry} if fallback_geometry is not None else fixture)
                            lane_scene,lane_spheres,_,_=prepare_photo_geometry(
                            {**config, 'camera': lane_camera, 'fixtures': lane_fixture,
                             'transport': lane_camera.get('sensor_transport', transport)},
                            vertices, material_ids, pbr, prepare_sources=False)
                            prepared_scene_cache['scenes'][lane_key]=(lane_scene,lane_spheres)
                        return prepared_scene_cache['scenes'][lane_key]
                    sweeps = max(1, int(round(dt * exposure_substeps / seconds)))
                    launched, submitted = _dispatch_timed_sensor_lanes(
                        lane_records, interval, interval_index,
                        prepare_scene=prepare_lane_scene, renderer=renderer,
                        barrier=timed_barrier, sweeps=sweeps, finish=glFinish,
                        geometry_revision=geometry_revision)
                    if any(lane.get('installed_geometry_revision')==geometry_revision and
                           lane.get('unsupported_geometry_revision')!=geometry_revision for lane in lane_records):
                        for lane in lane_records:
                            if lane.get('installed_geometry_revision')==geometry_revision:
                                lane['installed_fixture_geometry']=active_fixture_geometry
                    field_ray_jobs += launched
                    sensor_sites += submitted
                    epoch += 1
                    now = time.monotonic()
                    done = interval_index + 1 == len(timed_frames)
                    if done or now - last_snapshot >= snapshot_seconds:
                        lane_statuses = []
                        for lane in lane_records[1:]:
                            lane_output = output / f"lane-{lane['index']:02d}"
                            lane_status = _snapshot(lane['accumulator'], lane_output,
                                digest, epoch, now-job_start, preparation,
                                field_ray_jobs, sensor_sites, finished=done,
                                transport=lane['transport'], gamma=gamma,
                                film_response=film_response, display_params=display_params,
                                frame_elapsed=now-frame_start, publish_status=False,
                                monochrome=lane['camera'].get('sensor_output')=='mono',
                                calibration_regions=(config.get('calibration_regions') or [None] * len(lane_records))[lane['index']])
                            lane_statuses.append(dict(index=lane['index'],
                                camera=lane['camera'],
                                transport=lane['transport'],
                                sensor_sites=sensor_sites // len(lane_records),
                                image=f"{lane_output.name}/{lane_status['image']}",
                                marker=(f"{lane_output.name}/{lane_status['marker']}"
                                        if lane_status.get('marker') else None),
                                hdr=(f"{lane_output.name}/{lane_status['frames'][-1]['hdr']}"
                                     if done else None),
                                calibration_regions=lane_status.get('calibration_regions')))
                        status = _snapshot(accumulator, output, digest, epoch,
                            now-job_start, preparation, field_ray_jobs,
                            sensor_sites, finished=done,
                            transport=transport, gamma=gamma, film_response=film_response,
                            display_params=display_params,
                            frame_elapsed=now-frame_start,
                            calibration_regions=(config.get('calibration_regions') or [None] * len(lane_records))[0])
                        if lane_statuses:
                            status['sensor_lanes'] = [dict(index=0, camera=lane_records[0]['camera'],
                                transport=transport,
                                marker=status.get('marker'),
                                sensor_sites=sensor_sites // len(lane_records),
                                image=status['image'],
                                hdr=status['frames'][-1]['hdr'] if done else None), *lane_statuses]
                            status['sensor_lanes'][0]['calibration_regions'] = status.get('calibration_regions')
                        status['exposure_state_frames'] = interval_index + 1
                        status['stale_geometry_slices'] = sum(lane.get('stale_geometry_slices',0) for lane in lane_records)
                        status['exposure_integrated_seconds'] = float(interval['t1'])
                        if config.get('calibration_regions'):
                            calibration_history.append({
                                'epoch': epoch,
                                'exposure_integrated_seconds': float(interval['t1']),
                                'sensor_lanes': [
                                    {'index': item['index'],
                                     'calibration_regions': item.get('calibration_regions')}
                                    for item in status.get('sensor_lanes', [])]})
                            calibration_history = calibration_history[-128:]
                            status['calibration_history'] = calibration_history
                        _atomic_json(output/'status.json', status)
                        last_snapshot = now
                completed_frames = status['frames']
                for lane in lane_records:
                    lane['accumulator'].destroy()
                    lane['accumulator'] = None
                accumulator = None
                continue
            if tiles and work_index == 0:
                # Surface reception also samples the forward field. Pumping it
                # between tiles brightens later windows even after SPP division.
                # Every tile shares this exact existing 8192-ray prepared field.
                renderer._clear_prebuilt(state)
                renderer._gpu_pump_prebuilt(state, n_rays=8192)
                field_ray_jobs = 8192 if state.get('n_sources') else 0
                glFinish()
                tiles = {**tiles, "forward_field_ray_jobs": 8192,
                         "forward_field": "shared frozen after one prepared epoch"}
            frame_start = time.monotonic()
            last_snapshot = frame_start - snapshot_seconds
            frame_epochs = 0
            tile_sensor_sites = 0
            while True:
                if not tiles:
                    renderer._gpu_pump_prebuilt(state, n_rays=8192)
                    field_ray_jobs += 8192 if state.get('n_sources') else 0
                if cooperative_pump is not None:
                    # No forward field or camera mutation while a strip is on loan.
                    for _ in range(4):
                        cooperative_pump.step(accumulator)
                        time_done=time.monotonic()-frame_start>=seconds or (max_epochs is not None and frame_epochs+1>=max_epochs)
                        if time_done and accumulator._pass>=1 and accumulator._row_off==0:break
                else:
                    if tiles:
                        sites = accumulator._w * min(accumulator._rows_per_frame, accumulator._h - accumulator._row_off)
                        tile_sensor_sites += sites
                        tiled_sensor_sites += sites
                    if response_requested:
                        from camera_designer.studio_response_factory import rebind_response_owner
                        rebind_response_owner(accumulator,state,accumulator._frame_camera)
                    accumulator.tick(strips=1 if tiles else 4)
                epoch += 1
                frame_epochs += 1
                if epoch % 8 == 0:
                    # Do not queue an unbounded number of epochs ahead of the GPU.
                    glFinish()
                if not tiles and epoch % 4096 == 0:
                    # R32UI forward voxels must not overflow during long exposures.
                    glFinish()
                    seed = state["_pump_seed"]
                    renderer._clear_prebuilt(state)
                    state["_pump_seed"] = seed
                    field_ray_jobs = 0
                    if field_publisher is not None:
                        field_publisher.invalidate()
                now = time.monotonic()
                target_samples = tiles.get("target_samples_per_pixel") if tiles else None
                if target_samples is not None:
                    done = accumulator._pass * accumulator._samples_per_pixel >= target_samples
                    if max_epochs is not None and frame_epochs >= max_epochs and not done:
                        raise RuntimeError("Epoch limit reached before the uniform tile sample target")
                else:
                    done = (now - frame_start >= seconds or
                            (max_epochs is not None and frame_epochs >= max_epochs))
                if cooperative_pump is not None or tiles:done=done and accumulator._pass>=1 and accumulator._row_off==0
                if tiles and not done and now - last_snapshot >= snapshot_seconds:
                    # The UI receives queue progress, not a costly full-tile
                    # readback/tonemap that it cannot display as the final image.
                    status = {**preparation_status, "state": "exposing", "tiled": tiles,
                              "epoch": epoch, "elapsed_seconds": round(now-job_start,3),
                              "frame_elapsed_seconds": round(now-frame_start,3),
                              "forward_ray_jobs": 8192, "sensor_sites": tiled_sensor_sites,
                              "tile_completed_passes": accumulator._pass,
                              "tile_row": accumulator._row_off}
                    _atomic_json(output / "status.json", status)
                    print(json.dumps(status, sort_keys=True), flush=True)
                    last_snapshot = now
                if done or (not tiles and now - last_snapshot >= snapshot_seconds):
                    status = _snapshot(accumulator, work_output, digest, epoch,
                                       now - job_start, preparation,
                                       (8192 if tiles else epoch * 8192), (cooperative_pump.profile["committed_rays"] if cooperative_pump is not None else tiled_sensor_sites if tiles else frame_epochs * 4 * 16 * resolution),
                                       finished=done, transport=transport,
                                       frame_index=frame_index, total_frames=total_frames,
                                       frame_elapsed=now - frame_start,
                                       completed_frames=tuple(completed_frames),
                                       gamma=gamma, film_response=film_response,
                                       display_params=display_params,
                                       publish_status=field_publisher is None)
                    if field_publisher is not None:
                        exported = field_publisher.snapshot(renderer, state,
                            sweep=epoch, sensor_samples=accumulator._pass, ray_jobs=field_ray_jobs,
                            accumulation=('frozen shared field for all tiles' if tiles else
                                          'additive since last clear; existing 4096-epoch overflow reset'))
                        if exported is not None:
                            status['light_field'] = exported
                    if response_publisher is not None:
                        recorded=response_publisher.publish(response_consumer)
                        if recorded is not None:status['scene_responses']=recorded
                    if tiles:
                        # Tile previews are not the full image. Do not publish their
                        # URLs through the authenticated final-image route.
                        status = {**status, "state": "exposing", "image": None,
                                  "frames": [], "tiled": tiles}
                        status.pop("hdr", None)
                        _atomic_json(output / "status.json", status)
                    if cooperative_pump is not None:
                        status['client_work_profile']=dict(cooperative_pump.profile)
                        _atomic_json(output / 'status.json',status)
                    if field_publisher is not None and not tiles and cooperative_pump is None:
                        _atomic_json(output / 'status.json', status)
                    print(json.dumps(status, sort_keys=True), flush=True)
                    last_snapshot = now
                if done:
                    if tiles:
                        assembly.add(window, work_output / f"exposure-{epoch:06d}.npz",
                                     accumulator._pass * accumulator._samples_per_pixel)
                        tiles = {**tiles, "completed_tiles": work_index + 1, "active_tile": None}
                        _atomic_json(output / "status.json", {**status, "tiled": tiles})
                        # Persisted mean sensor storage is authoritative. Tile
                        # snapshots are temporary, not another image archive.
                        for artifact in work_output.iterdir():
                            artifact.unlink()
                        work_output.rmdir()
                    completed_frames = status["frames"]
                    accumulator.destroy()
                    accumulator = None
                    break
        if tiles:
            status = {**status, "state": "assembling", "tiled": tiles}
            _atomic_json(output / "status.json", status)
            artifacts = assembly.finish(film_response, gamma, geometry_stats)
            status = {**status, **artifacts, "state": "complete", "frame_index": 0,
                      "total_frames": 1, "tiled": tiles,
                      "elapsed_seconds": round(time.monotonic() - job_start, 3),
                      "frames": [{"index": 0, "image": artifacts["image"], "hdr": artifacts["hdr"]}]}
            _atomic_json(output / "status.json", status)
            print(json.dumps(status, sort_keys=True), flush=True)
        if total_frames > 1:
            if avi_writer_root is None:
                raise ValueError("camera sequence animation needs the Turing AVI writer root")
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
        if assembly is not None: assembly.close()
        if cooperative_pump is not None:cooperative_pump.close()
        if accumulator is not None:
            accumulator.destroy()
            if lane_records and lane_records[0].get('accumulator') is accumulator:
                lane_records[0]['accumulator'] = None
        for lane in lane_records:
            if lane.get('accumulator') is not None:
                lane['accumulator'].destroy()
            if lane['state'] is not state:
                renderer._destroy_prebuilt(lane['state'])
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
        status = {}
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            digest = hashlib.sha256(args.payload.read_bytes() +
                                    json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            previous = json.loads((args.out / "status.json").read_text(encoding="utf-8"))
            if isinstance(previous, dict) and previous.get("digest") == digest:
                status = previous
        except (OSError, ValueError, TypeError):
            pass
        _atomic_json(args.out / "status.json", {**status, "state": "error", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
