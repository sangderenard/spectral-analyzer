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
              frame_elapsed: float = 0.0, completed_frames: tuple[dict, ...] = (),
              gamma: float = 1.0, film_response: dict | None = None,
              display_params: dict | None = None, publish_status: bool = True) -> dict:
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
    if transport == "fixed:3":
        # FilmStack's ordered R/G/B emulsion layers are separate linear
        # RGBA32F exposure textures. Their RGB components repeat the scalar
        # exposure; read one component from each, preserving relative energy.
        layers = []
        for texture in accumulator._textures[:3]:
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
        linear = np.stack(layers, axis=2)
        # Preserve raw sensor values. Color characterization precedes all
        # preview tone/gamma operations and is never applied to the HDR twice.
        linear_output, color_profile = sensor_to_linear_srgb(linear, film_response)
        display, white = render_linear_srgb(linear_output, {"gamma": gamma, **(display_params or {})})
        luma = np.maximum(linear_output @ SRGB_LUMA, 0.0)
        mode = "RGB"
        hdr_name = f"exposure-{epoch:06d}.npz" if finished else None
        if finished:
            temporary_hdr = output / (hdr_name + ".tmp")
            with temporary_hdr.open("wb") as stream:
                response = (normalize_film_response() if film_response is None
                            else normalize_film_response(film_response))
                np.savez_compressed(
                    stream, linear_rgb_exposure=linear,
                    # Legacy discrete transport-band labels, not the shifted
                    # Gaussian centers or monochromatic channel measurements.
                    wavelengths_nm=np.asarray([650, 550, 450], np.float32),
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
                retouched_name = _render_retouch(linear_output, retouch_params, output, colorimetric=True)
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
        white = max(float(np.percentile(nonzero, 98)) if len(nonzero) else 0.0, 1e-6)
        display = np.clip(luma / white, 0, 1) ** inv_gamma
        display = np.uint8(display * 255)
        mode = "L"
        hdr_name = None
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
    frames = [*completed_frames]
    if finished:
        frames.append({"index": frame_index, "image": filename, "hdr": hdr_name})
    # Delete this epoch's now-superseded intermediate preview/HDR files.
    # Only the just-written frame and any already-"finished" frames (kept
    # for orbit playback) survive -- the progressive-preview history served
    # no purpose once a newer snapshot exists, and previously kept every
    # single intermediate render forever (confirmed: one long exposure left
    # 1,677 PNGs behind, 3.3 GiB, never cleaned up).
    keep = {filename}
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
        "transport": ("fixed:3@450,550,650nm" if transport == "fixed:3"
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
        status["hdr_channels"] = "raw linear sensor exposure plus separate profiled linear sRGB; one display transform"
    if color_profile is not None:
        status["color_profile"] = color_profile
        status["display_transform"] = display_transform_metadata(display_params, effective_white=white)
        if display_params is not None:
            status["live_display"] = display_params
        if retouched_name:
            status["retouch_params"] = retouch_params
            status["retouch_display_transform"] = display_transform_metadata(retouch_params,
                display_white_reference(linear_output, retouch_params))
        if color_profile["warning"]:
            status["color_profile_warning"] = color_profile["warning"]
    if film_response is not None:
        status["film_response"] = normalize_film_response(film_response)
        status["effective_config"] = "effective_config.json"
    if retouched_name:
        status["retouched_image"] = retouched_name
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


def run(payload: Path, config_path: Path, output: Path, *,
        seconds: float = 600.0, snapshot_seconds: float = 2.0,
        max_epochs: int | None = None, avi_writer_root: Path | None = None,
        dynamic_scene_receiver=None) -> dict:
    if not 0 < seconds <= 3600:
        raise ValueError("photo exposure per frame must be in (0, 3600] seconds")
    if not 0.25 <= snapshot_seconds <= 60:
        raise ValueError("snapshot interval must be in [0.25, 60] seconds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cooperative=config.get('client_intersections',False)
    if type(cooperative) is not bool:raise ValueError('client_intersections must be boolean')
    if cooperative and (config.get('transport')!='fixed:3' or config.get('orbit_frames',1)!=1):
        raise ValueError('Cooperative intersections currently require fixed:3 transport with one camera frame')
    max_bounces = _bounce_limit(config.get("max_bounces", 4))
    # Older direct worker jobs predate Auto; new Studio requests send the mode.
    bounce_mode = config.get("bounce_mode", "manual")
    if bounce_mode not in ("auto", "manual"):
        raise ValueError("bounce_mode must be auto or manual")
    total_frames = int(config.get("orbit_frames", 1))
    if total_frames not in (1, 8, 16, 32):
        raise ValueError("orbit_frames must be 1, 8, 16, or 32")
    transport = str(config.get("transport", "fixed:1"))
    if transport not in ("fixed:1", "fixed:3"):
        raise ValueError("transport must be fixed:1 or fixed:3")
    if transport == "fixed:1" and "film_response" in config:
        raise ValueError("film_response requires fixed:3 RGB transport")
    film_response = None
    if transport == "fixed:3":
        film_response = (normalize_film_response(config["film_response"])
                         if "film_response" in config else normalize_film_response())
    gamma = float(config.get("gamma", 1.0))
    if not 0.2 <= gamma <= 5.0:
        raise ValueError("gamma must be in [0.2, 5.0]")
    wavelengths = [.5875618] if transport == "fixed:1" else [.450, .550, .650]
    digest = hashlib.sha256(payload.read_bytes() +
                            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the caller's digest/config identity, while recording defaults
    # explicitly for reproducibility of old requests that omit this field.
    effective_config = dict(config)
    if film_response is not None:
        effective_config["film_response"] = film_response
    _atomic_json(output / "effective_config.json", effective_config)
    preparation_status = {
        "digest": digest, "state": "preparing", "image": None,
        "frames": [], "frame_index": 0, "total_frames": total_frames,
        "transport": ("fixed:3@450,550,650nm" if transport == "fixed:3"
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

    preparation_start = time.monotonic()
    from camera_designer.studio_photo_geometry import (
        CAMERA_GEOMETRY_VERSION, camera_geometry_key, prepare_photo_geometry,
    )
    scene, sphere_regions, geometry_stats, checker_pos = prepare_photo_geometry(config, vertices, material_ids, pbr)
    geom, shade, mat_buf, triangles, contexts, _, bounds, sources = scene
    preparation_status["geometry_stats"] = geometry_stats
    effective_config["geometry_stats"] = geometry_stats
    effective_config["camera_geometry_version"] = CAMERA_GEOMETRY_VERSION
    _atomic_json(output / "effective_config.json", effective_config)
    _atomic_json(output / "status.json", preparation_status)
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
    from material_db import MaterialDatabase
    state = renderer._gpu_ray_field_prebuilt(
        geom, shade, mat_buf, nodes, ids, contexts, sources, bounds,
        dims=(96, 96, 96), dispatch_batch=512, max_bounces=max_bounces, initial_rays=128,
        film_stack=film, sphere_regions=sphere_regions, region_records=region_records,
        native_pbr=MaterialDatabase.instance().build_tensors()["pbr"],
    )
    if state is None:
        raise RuntimeError("Spectral GPU ray-field shader did not link")
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
    _atomic_json(output / "effective_config.json", effective_config)
    _atomic_json(output / "status.json", {**preparation_status,
        "bounce_diagnostic": bounce_diagnostic, "effective_config": "effective_config.json"})
    print(f"  [marble_bounces] {bounce_mode}: render={max_bounces}/200; "
          f"probe={bounce_diagnostic.get('probe_depth', 'not requested')}; "
          f"{bounce_diagnostic.get('warning', '')}", flush=True)
    camera = config.get("camera", {})
    resolution = int(camera.get("resolution", 256))
    if resolution not in (128, 256, 512, 768, 1024, 2048, 4096):
        raise ValueError("preview resolution must be 128, 256, 512, 768, 1024, 2048, or 4096")
    live = config.get("live")
    width = int(live["width"]) if live else resolution
    height = int(live["height"]) if live else resolution
    if live:
        from camera_designer.sensor_history import SensorHistoryLayout
        if transport != "fixed:3" or total_frames != 1 or width > 512 or height > 512:
            raise ValueError("live view requires a bounded RGB single-camera sensor")
        SensorHistoryLayout(width, height, 3, live["history_samples"], live.get("checkpoint", False))
    eye = np.asarray(camera.get("eye_m", [0, 0, .07]), np.float32)
    target = np.asarray(camera.get("target_m", checker_pos), np.float32)
    active_camera_key = camera_geometry_key(camera, checker_pos)

    def update_camera_geometry(next_camera, active_sensor=None):
        nonlocal active_camera_key, geometry_stats, triangles, nodes, sphere_regions
        next_key = camera_geometry_key(next_camera, checker_pos)
        if next_key == active_camera_key:
            return
        next_config = {**config, "camera": next_camera}
        next_scene, next_spheres, next_stats, _ = prepare_photo_geometry(
            next_config, vertices, material_ids, pbr)
        next_geom, next_shade, next_mat, next_tris, next_ctx, _, next_bounds, _ = next_scene
        next_bounds = (next_bounds[0] - margin, next_bounds[1] + margin)
        next_nodes, next_ids = renderer._build_gpu_bvh(next_tris)
        renderer._update_prebuilt_geometry(state, next_geom, next_shade, next_mat,
            next_nodes, next_ids, next_ctx, next_bounds, next_spheres,
            native_pbr=MaterialDatabase.instance().build_tensors()["pbr"])
        if active_sensor is not None:
            active_sensor.set_scene_geometry(n_tris=len(next_tris), n_bvh_nodes=len(next_nodes),
                bmin=next_bounds[0], bmax=next_bounds[1], n_spheres=len(next_spheres))
            active_sensor._geometry_stats = next_stats
        geometry_stats, triangles, nodes, sphere_regions = next_stats, next_tris, next_nodes, next_spheres
        active_camera_key = next_key
        # Old-pose photons cannot survive into an orbit frame or live sweep.
        # Preserve stochastic sequence, but never mix two different scenes.
        seed = state["_pump_seed"]
        renderer._clear_prebuilt(state)
        state["_pump_seed"] = seed

    def sensor(frame_index: int):
        angle = 2.0 * np.pi * frame_index / total_frames
        dx, dy, dz = eye - target
        camera_eye = target + np.asarray((np.cos(angle) * dx + np.sin(angle) * dz,
                                          dy, -np.sin(angle) * dx + np.cos(angle) * dz), np.float32)
        update_camera_geometry({**camera, "eye_m": camera_eye.tolist(), "target_m": target.tolist()})
        device_limits=_sensor_device_limits(width,height)
        print(json.dumps({"sensor_device_limits":device_limits}),flush=True)
        accumulator = renderer.SensorAccumulator(
            ssbo_tris=state["ssbo"][0], ssbo_nodes=state["ssbo"][1],
            ssbo_ids=state["ssbo"][2], ssbo_shade=state["ssbo"][9],
            ssbo_mat=state["ssbo"][10], n_tris=len(triangles),
            n_bvh_nodes=len(nodes), tex_bands=state["tex_bands"],
            bmin=state["bmin"], bmax=state["bmax"], dims=state["dims"],
            camera_eye=camera_eye, camera_target=target,
            sensor_w=width, sensor_h=height, rows_per_frame=(max(1,4096//width) if cooperative and not live else 16),
            fov_deg=float(camera.get("fov_degrees", 20)), max_bounces=max_bounces,
            aperture_radius=float(camera.get("aperture_radius_m", .005)),
            focus_dist=float(camera.get("focus_distance_m", .23)),
            n_blades=int(camera.get("aperture_blades", 0)),
            aperture_rot=float(camera.get("aperture_rotation_radians", 0)),
            vol_alpha=0.0, vol_steps=8,
            n_sources=len(sources),
            ssbo_spheres=state["ssbo"][12], n_spheres=len(sphere_regions),
            ssbo_regions=state["ssbo"][15] if state.get("region_count") else 0,
            n_regions=state.get("region_count",0),
            ssbo_native_pbr=state["native_pbr_ssbo"],
        )
        if not accumulator._active:
            accumulator.destroy()
            raise RuntimeError("Spectral sensor compute shader did not link")
        if film is not None:
            accumulator.film = film
        if transport == "fixed:3" or geometry_stats.get('emissive_sampling'):
            try:
                accumulator.set_nee_spectral_sources(sources, state["ssbo"][5],
                    area_scene=scene if geometry_stats.get('emissive_sampling') else None)
            except Exception:
                accumulator.destroy()
                raise
        accumulator._has_rod_bundle = authored_bundle is not None
        accumulator._geometry_stats = geometry_stats
        accumulator._bounce_diagnostic = bounce_diagnostic
        accumulator.half_life = 0.0
        accumulator._device_limits=device_limits
        accumulator._sprinkle = False  # existing deterministic row-priority mode
        return accumulator
    preparation = time.monotonic() - preparation_start
    epoch = 0
    completed_frames: list[dict] = []
    accumulator = None
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
            return run_live_sensor(accumulator, renderer, state, output, digest,
                                   config, preparation, _snapshot, _atomic_json,
                                   max_epochs=max_epochs,
                                   update_camera_geometry=update_camera_geometry,
                                   dynamic_scene_reader=dynamic_reader)
        for frame_index in range(total_frames):
            accumulator = sensor(frame_index)
            frame_start = time.monotonic()
            last_snapshot = frame_start - snapshot_seconds
            frame_epochs = 0
            while True:
                renderer._gpu_pump_prebuilt(state, n_rays=8192)
                if cooperative_pump is not None:
                    # No forward field or camera mutation while a strip is on loan.
                    for _ in range(4):
                        cooperative_pump.step(accumulator)
                        time_done=time.monotonic()-frame_start>=seconds or (max_epochs is not None and frame_epochs+1>=max_epochs)
                        if time_done and accumulator._pass>=1 and accumulator._row_off==0:break
                else:accumulator.tick(strips=4)
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
                if cooperative_pump is not None:done=done and accumulator._pass>=1 and accumulator._row_off==0
                if now - last_snapshot >= snapshot_seconds or done:
                    status = _snapshot(accumulator, output, digest, epoch,
                                       now - job_start, preparation,
                                       epoch * 8192, (cooperative_pump.profile["committed_rays"] if cooperative_pump is not None else frame_epochs * 4 * 16 * resolution),
                                       finished=done, transport=transport,
                                       frame_index=frame_index, total_frames=total_frames,
                                       frame_elapsed=now - frame_start,
                                       completed_frames=tuple(completed_frames),
                                       gamma=gamma, film_response=film_response)
                    if cooperative_pump is not None:
                        status['client_work_profile']=dict(cooperative_pump.profile)
                        _atomic_json(output / 'status.json',status)
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
        if cooperative_pump is not None:cooperative_pump.close()
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
