"""Editable paragraph UI backed by asynchronous spectral thick-lens renders."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from camera_software import ColorScienceProfile, save_linear_sensor_image


DEFAULT_TEXT = "Actual light takes the long way home through glass."
JOB_ID = "live_paragraph"
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 600
SENSOR_CROP_SCALE = 5


def render_contract_summary(
    display_width: int,
    display_height: int,
    sensor_sweeps: int = 1,
) -> str:
    """Describe requested output and inherited renderer work without guessing runtime sampling."""
    width = int(display_width)
    height = int(display_height)
    native_resolution = max(width, height)
    return (
        f"output={width}x{height} native_sensor={native_resolution}x{native_resolution} "
        f"composition_frame={width * SENSOR_CROP_SCALE}x{height * SENSOR_CROP_SCALE} "
        f"crop=({2 * width},{2 * height},{width},{height}) "
        f"sensor_sweeps={int(sensor_sweeps)}; "
        "composition_frame is coordinates only, not a rendered raster"
    )


def build_paragraph_order(
    text: str,
    *,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    sensor_sweeps: int = 1,
) -> dict[str, Any]:
    """Build a paragraph order sampled at the digital display's exact raster."""
    token = str(text).strip()
    if not token:
        token = " "
    output_width = int(display_width)
    output_height = int(display_height)
    if output_width <= 0 or output_height <= 0:
        raise ValueError("display dimensions must be positive")
    if int(sensor_sweeps) <= 0:
        raise ValueError("sensor_sweeps must be positive")
    full_width = output_width * SENSOR_CROP_SCALE
    full_height = output_height * SENSOR_CROP_SCALE
    plane_normal = [0.8017837257, -0.5345224838, 0.2672612419]
    exposed_text_depth_m = 0.012 * (1.0 - 0.25)
    focus_target = [component * exposed_text_depth_m for component in plane_normal]
    return {
        "schema_version": 1,
        "defaults": {
            "image": {
                "width": full_width,
                "height": full_height,
                "region": {
                    "x": 2 * output_width,
                    "y": 2 * output_height,
                    "width": output_width,
                    "height": output_height,
                },
            },
            "camera": {
                "focal_mm": 35.0,
                "aperture_mm": 25.0,
                "position_m": [3.0, -2.0, 1.0],
                "target_m": [0.0, 0.0, 0.0],
                "focus_target_m": focus_target,
                "up": [0.0, 0.0, 1.0],
            },
            "exposure": {
                "time_s": 0.016666666666666666,
                "iso": 100.0,
                "sensor_sweeps": int(sensor_sweeps),
                "t5_pair_budget": 20000000,
            },
            "flash": {"intensity_scale": 1.0},
            "font": {
                "family": "DejaVu Sans",
                "weight": "bold",
                "style": "normal",
            },
            "planes": [{
                "id": "background",
                "center_m": [0.0, 0.0, 0.0],
                "normal": plane_normal,
                "up": [-0.2223747950, 0.1482498633, 0.9636241117],
                "size_m": [1.4, 1.0],
                "thickness_m": 0.018,
                "material": "quiet_background",
            }],
            "materials": {
                "quiet_background": {
                    "albedo_rgb": [0.018, 0.021, 0.022],
                    "reflectivity": 0.065,
                    "diffusion": 0.98,
                    "absorption": 0.92,
                    "roughness": 0.98,
                    "metallic": 0.0,
                },
                "text_surface": {
                    "albedo_rgb": [0.82, 0.76, 0.62],
                    "reflectivity": 0.58,
                    "diffusion": 0.72,
                    "absorption": 0.25,
                    "roughness": 0.3,
                    "metallic": 0.0,
                    "ior": 1.5,
                },
            },
            "geometry": {
                "embed_plane": "background",
                "height_m": 0.055,
                "line_height_m": 0.055,
                "line_spacing": 1.22,
                "text_box_m": [0.31, 0.18],
                "horizontal_align": "center",
                "vertical_align": "center",
                "depth_m": 0.012,
                "embed_fraction": 0.25,
                "offset_m": [0.0, 0.0],
                "profile": "straight",
                "outline_subdivisions": 1,
                "cap_grid": 34,
                "material": "text_surface",
            },
        },
        "jobs": [{"id": JOB_ID, "token": token}],
    }


@dataclass(frozen=True)
class RenderedTextRevision:
    sequence: int
    text: str
    image_path: str
    linear_path: str
    manifest_path: str
    elapsed_s: float
    diagnostic_image_path: str = ""


RenderFunction = Callable[[int, str, dict[str, Any]], RenderedTextRevision]


class SpectralTextRenderWorker:
    """Single render owner with a coalescing one-item pending slot."""

    def __init__(self, render_function: RenderFunction):
        self._render_function = render_function
        self._condition = threading.Condition()
        self._pending: tuple[int, str, dict[str, Any]] | None = None
        self._latest: RenderedTextRevision | None = None
        self._latest_sequence = 0
        self._submitted_sequence = 0
        self._busy = False
        self._error = ""
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="spectral-text-render-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, text: str, order: dict[str, Any]) -> int:
        with self._condition:
            self._submitted_sequence += 1
            sequence = self._submitted_sequence
            self._pending = (sequence, str(text), copy.deepcopy(order))
            self._condition.notify()
            return sequence

    def snapshot(self) -> tuple[RenderedTextRevision | None, bool, str, int]:
        with self._condition:
            return self._latest, self._busy, self._error, self._submitted_sequence

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                sequence, text, order = self._pending
                self._pending = None
                self._busy = True
                self._error = ""
            try:
                result = self._render_function(sequence, text, order)
                with self._condition:
                    self._latest = result
                    self._latest_sequence = sequence
            except Exception as exc:
                with self._condition:
                    self._error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()


def make_subprocess_renderer(
    output_root: str,
    camera_profile: ColorScienceProfile,
) -> RenderFunction:
    root = os.path.abspath(output_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def render(sequence: int, text: str, order: dict[str, Any]) -> RenderedTextRevision:
        revision_dir = os.path.join(root, f"revision_{sequence:04d}")
        os.makedirs(revision_dir, exist_ok=True)
        order_path = os.path.join(revision_dir, "scene_order.json")
        with open(order_path, "w", encoding="utf-8") as handle:
            json.dump(order, handle, indent=2)
        started = time.perf_counter()
        command = [
            sys.executable,
            os.path.join(script_dir, "render_scene_order.py"),
            order_path,
            "--render",
            "--out-dir",
            revision_dir,
        ]
        log_path = os.path.join(revision_dir, "render.log")
        print(f"[live] revision {sequence} process: {' '.join(command)}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[render {sequence:04d}] {line}", end="", flush=True)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        job_dir = os.path.join(revision_dir, JOB_ID)
        linear_path = os.path.join(job_dir, "0000_cpp_linear.npy")
        diagnostic_image_path = os.path.join(job_dir, "0000_cpp.png")
        camera_image_path = os.path.join(job_dir, "0000_cpp_camera.png")
        linear_sensor = np.load(linear_path, allow_pickle=False)
        save_linear_sensor_image(linear_sensor, camera_image_path, camera_profile)
        print(
            f"[live] camera output revision {sequence}: raw={linear_path} "
            f"display={camera_image_path} white_level={camera_profile.sensor_white_level:g} "
            f"exposure_ev={camera_profile.exposure_compensation_ev:+g} "
            f"tone={camera_profile.tone_curve_mode}",
            flush=True,
        )
        return RenderedTextRevision(
            sequence=sequence,
            text=text,
            image_path=camera_image_path,
            linear_path=linear_path,
            manifest_path=os.path.join(job_dir, "composition_manifest.json"),
            elapsed_s=time.perf_counter() - started,
            diagnostic_image_path=diagnostic_image_path,
        )

    return render


def _wrapped_editor_lines(font: Any, text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _texture_display_rect(
    source_width: int,
    source_height: int,
    window_width: int,
    window_height: int,
) -> tuple[int, int, int, int]:
    """Fit a texture into the window without enlarging traced pixels."""
    scale = min(
        1.0,
        float(window_width) / float(source_width),
        float(window_height) / float(source_height),
    )
    width = max(1, int(round(float(source_width) * scale)))
    height = max(1, int(round(float(source_height) * scale)))
    return (window_width - width) // 2, (window_height - height) // 2, width, height


def _hud_layout(width: int) -> dict[str, int]:
    """Size a separate editor strip for the available display width."""
    scale = min(1.0, max(0.4, float(width) / 480.0))
    editor_font_px = max(10, int(round(24.0 * scale)))
    status_font_px = max(9, int(round(20.0 * scale)))
    padding = max(4, int(round(14.0 * scale)))
    editor_lines = 2 if width < 320 else 4
    editor_line_height = max(12, int(round(editor_font_px * 1.2)))
    status_line_height = max(11, int(round(status_font_px * 1.2)))
    height = padding * 3 + editor_lines * editor_line_height + status_line_height
    return {
        "height": height,
        "padding": padding,
        "editor_font_px": editor_font_px,
        "status_font_px": status_font_px,
        "editor_lines": editor_lines,
    }


def _window_regions(width: int, height: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Partition the window into non-overlapping canvas and HUD regions."""
    hud_height = min(_hud_layout(width)["height"], max(0, height - 1))
    canvas_height = height - hud_height
    return (0, 0, width, canvas_height), (0, canvas_height, width, hud_height)


def run_window(
    output_root: str,
    initial_text: str,
    display_width: int,
    display_height: int,
    sensor_sweeps: int,
    camera_profile: ColorScienceProfile,
) -> int:
    import pygame

    print(
        f"[live] {render_contract_summary(display_width, display_height, sensor_sweeps)}",
        flush=True,
    )
    print(
        "[live] renderer policy is inherited unchanged; actual aperture samples, "
        "camera-ray schedule, T5 work, and convergence mode will stream below",
        flush=True,
    )
    print(
        "[live] camera output "
        f"white_level={camera_profile.sensor_white_level:g} "
        f"exposure_ev={camera_profile.exposure_compensation_ev:+g} "
        f"white_balance={np.asarray(camera_profile.white_balance).tolist()} "
        f"tone={camera_profile.tone_curve_mode} output={camera_profile.output_space}",
        flush=True,
    )
    pygame.init()
    pygame.key.start_text_input()
    initial_hud = _hud_layout(display_width)
    window = pygame.display.set_mode(
        (display_width, display_height + initial_hud["height"]),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption("Spectral text surface")
    clock = pygame.time.Clock()

    worker = SpectralTextRenderWorker(
        make_subprocess_renderer(output_root, camera_profile)
    )
    text = str(initial_text)[:500]
    cursor = len(text)
    submitted_text = ""
    dirty_at = time.monotonic() - 2.0
    loaded_sequence = 0
    texture = None
    font_sizes: tuple[int, int] | None = None
    font = None
    status_font = None
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.TEXTINPUT:
                    text = (text[:cursor] + event.text + text[cursor:])[:500]
                    cursor = min(len(text), cursor + len(event.text))
                    dirty_at = time.monotonic()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_BACKSPACE and cursor > 0:
                        text = text[:cursor - 1] + text[cursor:]
                        cursor -= 1
                        dirty_at = time.monotonic()
                    elif event.key == pygame.K_DELETE and cursor < len(text):
                        text = text[:cursor] + text[cursor + 1:]
                        dirty_at = time.monotonic()
                    elif event.key == pygame.K_LEFT:
                        cursor = max(0, cursor - 1)
                    elif event.key == pygame.K_RIGHT:
                        cursor = min(len(text), cursor + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        text = text[:cursor] + "\n" + text[cursor:]
                        cursor += 1
                        dirty_at = time.monotonic()

            if text.strip() and text != submitted_text and time.monotonic() - dirty_at >= 1.0:
                sequence = worker.submit(text, build_paragraph_order(
                    text,
                    display_width=display_width,
                    display_height=display_height,
                    sensor_sweeps=sensor_sweeps,
                ))
                print(
                    f"[live] submitted revision {sequence}: characters={len(text)} "
                    f"{render_contract_summary(display_width, display_height, sensor_sweeps)}",
                    flush=True,
                )
                submitted_text = text

            latest, busy, error, submitted_sequence = worker.snapshot()
            if latest is not None and latest.sequence != loaded_sequence:
                texture = pygame.image.load(latest.image_path).convert()
                loaded_sequence = latest.sequence
                print(
                    f"[live] completed revision {latest.sequence}: "
                    f"texture={texture.get_width()}x{texture.get_height()} "
                    f"elapsed={latest.elapsed_s:.3f}s image={latest.image_path}",
                    flush=True,
                )

            width, height = window.get_size()
            canvas_rect, hud_rect = _window_regions(width, height)
            hud = _hud_layout(width)
            requested_font_sizes = (hud["editor_font_px"], hud["status_font_px"])
            if requested_font_sizes != font_sizes:
                font = pygame.font.Font(None, requested_font_sizes[0])
                status_font = pygame.font.Font(None, requested_font_sizes[1])
                font_sizes = requested_font_sizes
            window.fill((15, 18, 19))
            if texture is None:
                pass
            else:
                texture_rect = _texture_display_rect(
                    texture.get_width(), texture.get_height(),
                    canvas_rect[2], canvas_rect[3],
                )
                draw_texture = texture
                if texture_rect[2:] != texture.get_size():
                    draw_texture = pygame.transform.smoothscale(texture, texture_rect[2:])
                window.blit(draw_texture, (
                    canvas_rect[0] + texture_rect[0],
                    canvas_rect[1] + texture_rect[1],
                ))

            overlay = pygame.Surface(hud_rect[2:], pygame.SRCALPHA)
            overlay.fill((8, 10, 11, 255))
            padding = hud["padding"]
            lines = _wrapped_editor_lines(font, text, max(1, width - 2 * padding))
            line_y = padding
            for line in lines[-hud["editor_lines"]:]:
                overlay.blit(font.render(line, True, (232, 229, 215)), (padding, line_y))
                line_y += font.get_linesize()
            if error:
                status = error if width >= 320 else error.split(":", 1)[0]
                status_color = (238, 116, 100)
            elif busy:
                status = (
                    f"Tracing revision {submitted_sequence}"
                    if width >= 320 else f"Tracing {submitted_sequence}"
                )
                status_color = (226, 188, 102)
            elif latest is not None:
                if width >= 320:
                    status = (
                        f"Revision {latest.sequence}: {display_width}x{display_height} source, "
                        f"completed in {latest.elapsed_s:.1f}s"
                    )
                else:
                    status = f"Done {latest.sequence}: {latest.elapsed_s:.1f}s"
                status_color = (124, 202, 162)
            else:
                status = "Waiting for first exposure" if width >= 320 else "Waiting"
                status_color = (170, 178, 178)
            status_surface = status_font.render(status, True, status_color)
            status_width = max(1, width - 2 * padding)
            if status_surface.get_width() > status_width:
                status_surface = pygame.transform.smoothscale(
                    status_surface,
                    (status_width, max(1, int(status_surface.get_height() * status_width / status_surface.get_width()))),
                )
            overlay.blit(status_surface, (
                padding,
                hud_rect[3] - status_surface.get_height() - padding,
            ))
            window.blit(overlay, hud_rect[:2])
            pygame.display.flip()
            clock.tick(60)
    finally:
        worker.close()
        pygame.key.stop_text_input()
        pygame.quit()
    return 0


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="exposures/live_spectral_text")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH,
                        help="Horizontal pixels in each ray-traced display texture.")
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT,
                        help="Vertical pixels in each ray-traced display texture.")
    parser.add_argument("--sensor-sweeps", type=int, default=1,
                        help="Independent full-sensor sweeps averaged per revision.")
    parser.add_argument("--camera-white-level", type=float, default=1.0,
                        help="Fixed linear sensor value mapped as camera white (default: 1).")
    parser.add_argument("--camera-exposure-ev", type=float, default=0.0,
                        help="Camera output exposure compensation in stops (default: 0).")
    parser.add_argument("--camera-white-balance", type=float, nargs=3,
                        metavar=("R", "G", "B"), default=(1.0, 1.0, 1.0),
                        help="Fixed camera RGB white-balance gains (default: 1 1 1).")
    parser.add_argument("--camera-tone-curve", choices=("linear", "reinhard"),
                        default="reinhard",
                        help="Fixed camera output tone curve (default: reinhard).")
    parser.add_argument("--write-order", metavar="PATH",
                        help="Write the resolved demo order and exit without opening a window.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(sys.argv[1:] if argv is None else argv)
    camera_profile = ColorScienceProfile.spectral_sensor_srgb(
        sensor_white_level=args.camera_white_level,
        exposure_compensation_ev=args.camera_exposure_ev,
        white_balance=np.asarray(args.camera_white_balance, dtype=np.float64),
        tone_curve_mode=args.camera_tone_curve,
    )
    if args.write_order:
        with open(args.write_order, "w", encoding="utf-8") as handle:
            json.dump(build_paragraph_order(
                args.text,
                display_width=args.display_width,
                display_height=args.display_height,
                sensor_sweeps=args.sensor_sweeps,
            ), handle, indent=2)
        return 0
    return run_window(
        args.out_dir,
        args.text,
        args.display_width,
        args.display_height,
        args.sensor_sweeps,
        camera_profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
