"""Small public triangle-mesh viewer for Pluck's ordinary OpenGL renderer.

This adapter owns window, VAO, and camera mechanics while delegating material
shading to :class:`base_gl_renderer.BaseGLRenderer`.  Numerical projects can
therefore present a triangle soup in Pluck without importing the full game.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LiveMeshFrame:
    triangles: np.ndarray
    triangle_values: np.ndarray | None
    panel_lines: Sequence[str]
    time_value: float
    published_at: float = field(default_factory=perf_counter)


def rolling_profile_lines(
    current: Mapping[str, float],
    history: Sequence[Mapping[str, float]],
    *,
    time_value: float | None = None,
) -> list[str]:
    """Format current and rolling timings without suppressing warm-up runs."""
    names = list(current)
    lines = []
    if time_value is not None:
        lines.append(f"simulation t       {time_value:8.4f}")
    lines.extend(("", "stage                    now      mean       p95"))
    for name in names:
        values = np.asarray(
            [row[name] for row in history if name in row], dtype=np.float64
        )
        p95 = float(np.quantile(values, 0.95)) if len(values) else np.nan
        mean = float(values.mean()) if len(values) else np.nan
        lines.append(
            f"{name:<22} {current[name]*1e3:7.1f} "
            f"{mean*1e3:8.1f} {p95*1e3:8.1f} ms"
        )
    lines.extend((
        "",
        f"runs included      {len(history):8d}",
        "warm-up is included; wall clock",
        "times cover complete stage calls",
    ))
    return lines


def summarize_video_profile(
    history: Mapping[str, Sequence[float]],
    *,
    rendered_frames: int,
    session_elapsed_sec: float,
) -> dict:
    """Return serializable CPU video timing statistics."""
    stages = {}
    for name, values in history.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array):
            stages[name] = {
                "count": int(len(array)),
                "mean_sec": float(array.mean()),
                "p95_sec": float(np.quantile(array, 0.95)),
                "max_sec": float(array.max()),
            }
    return {
        "rendered_frames": int(rendered_frames),
        "session_elapsed_sec": float(session_elapsed_sec),
        "stages": stages,
        "gpu_completion_forced": False,
    }


def triangle_mesh_vertex_rows(triangles: np.ndarray) -> np.ndarray:
    """Return Pluck vertex rows ``[position, flat normal, uv]``."""
    mesh = np.asarray(triangles, dtype=np.float32)
    if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N, 3, 3)")
    if mesh.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    edge_a = mesh[:, 1] - mesh[:, 0]
    edge_b = mesh[:, 2] - mesh[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths > 1e-12, lengths, 1.0)
    normal_rows = np.repeat(normals, 3, axis=0)
    positions = mesh.reshape(-1, 3)
    uv = np.zeros((positions.shape[0], 2), dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate((positions, normal_rows, uv), axis=1), dtype=np.float32
    )


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y * 0.5)
    matrix = np.zeros((4, 4), np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = 2.0 * far * near / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward) or 1.0
    up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) or 1.0
    camera_up = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def scalar_triangle_bins(
    values: np.ndarray, *, bin_count: int = 33, limit: float | None = None
) -> tuple[np.ndarray, float]:
    """Map signed triangle scalars to symmetric, robust palette bins."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("triangle values must be a vector")
    if bin_count < 3 or bin_count % 2 == 0:
        raise ValueError("bin_count must be an odd integer of at least 3")
    finite = np.abs(values[np.isfinite(values)])
    if limit is None:
        limit = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
    limit = max(float(limit), np.finfo(np.float64).eps)
    normalized = np.clip(values / limit, -1.0, 1.0)
    bins = np.rint((normalized + 1.0) * 0.5 * (bin_count - 1))
    bins = np.where(np.isfinite(bins), bins, bin_count // 2)
    return bins.astype(np.int32), limit


def _diverging_color(position: float) -> list[float]:
    neutral = np.asarray((0.78, 0.80, 0.77))
    endpoint = (
        np.asarray((0.12, 0.38, 0.96))
        if position < 0.0
        else np.asarray((0.96, 0.23, 0.10))
    )
    return ((1.0 - abs(position)) * neutral + abs(position) * endpoint).tolist()


def render_triangle_mesh_image(
    triangles: np.ndarray,
    output_path: str | Path,
    *,
    triangle_values: np.ndarray | None = None,
    value_label: str = "scalar value",
    title: str = "Pluck triangle mesh",
    size: tuple[int, int] = (1400, 1000),
    elevation: float = 24.0,
    azimuth: float = -58.0,
    side_panel_lines: Sequence[str] | None = None,
) -> Path:
    """Software-rasterize a deterministic headless mesh snapshot."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = np.asarray(triangles, dtype=np.float64)
    triangle_mesh_vertex_rows(mesh)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dpi = 100
    figure = plt.figure(
        figsize=(size[0] / dpi, size[1] / dpi),
        dpi=dpi,
        facecolor=(0.012, 0.018, 0.03),
    )
    if side_panel_lines:
        grid = figure.add_gridspec(1, 2, width_ratios=(2.45, 1.55))
        axes = figure.add_subplot(grid[0, 0], projection="3d")
        panel = figure.add_subplot(grid[0, 1])
        panel.set_facecolor((0.025, 0.035, 0.055))
        panel.set_axis_off()
        panel.text(
            0.06,
            0.96,
            "\n".join(side_panel_lines),
            va="top",
            ha="left",
            color=(0.82, 0.9, 0.98),
            family="monospace",
            fontsize=8.5,
            transform=panel.transAxes,
        )
    else:
        axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor((0.012, 0.018, 0.03))
    if triangle_values is None:
        colors = np.repeat(
            np.asarray(((0.18, 0.56, 0.92, 1.0),)), len(mesh), axis=0
        )
        color_limit = None
    else:
        bins, color_limit = scalar_triangle_bins(triangle_values)
        colors = np.asarray(
            [(*_diverging_color(2.0 * index / 32.0 - 1.0), 1.0) for index in bins]
        )
    collection = Poly3DCollection(
        mesh,
        facecolors=colors,
        edgecolors=(0.02, 0.025, 0.04, 0.24),
        linewidths=0.25,
    )
    axes.add_collection3d(collection)
    flattened = mesh.reshape(-1, 3)
    lower = flattened.min(axis=0)
    upper = flattened.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-6)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1, 1, 1))
    axes.view_init(elev=elevation, azim=azimuth)
    axes.set_axis_off()
    caption = title
    if color_limit is not None:
        caption += (
            f"\n{value_label}  |  blue −{color_limit:.3g}   neutral 0"
            f"   red +{color_limit:.3g}"
        )
    axes.set_title(caption, color="white", pad=16)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0.02)
    figure.savefig(output, dpi=dpi, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output.resolve()


def _upload_mesh(
    rows: np.ndarray, material_id: int | np.ndarray
) -> Tuple[int, ...]:
    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_FALSE,
        GL_FLOAT,
        GL_INT,
        GL_STATIC_DRAW,
        glBindBuffer,
        glBindVertexArray,
        glBufferData,
        glEnableVertexAttribArray,
        glGenBuffers,
        glGenVertexArrays,
        glVertexAttribIPointer,
        glVertexAttribPointer,
    )

    count = rows.shape[0]
    vao = int(glGenVertexArrays(1))
    buffers = tuple(int(x) for x in glGenBuffers(5))
    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    glBufferData(GL_ARRAY_BUFFER, rows.nbytes, rows, GL_STATIC_DRAW)
    stride = rows.shape[1] * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

    material_ids = np.asarray(material_id, dtype=np.int32)
    if material_ids.ndim == 0:
        material_ids = np.full(count, int(material_ids), np.int32)
    if material_ids.shape != (count,):
        raise ValueError("material IDs need one value per vertex")
    integer_columns = (
        material_ids,
        np.zeros(count, np.int32),
        np.ones(count, np.int32),
    )
    for buffer_id, location, values in zip(
        buffers[1:4], (2, 4, 5), integer_columns
    ):
        glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
        glBufferData(GL_ARRAY_BUFFER, values.nbytes, values, GL_STATIC_DRAW)
        glEnableVertexAttribArray(location)
        glVertexAttribIPointer(location, 1, GL_INT, 4, ctypes.c_void_p(0))

    velocity = np.zeros((count, 3), np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, buffers[4])
    glBufferData(GL_ARRAY_BUFFER, velocity.nbytes, velocity, GL_STATIC_DRAW)
    glEnableVertexAttribArray(6)
    glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)
    return (vao, *buffers)


class _OpenGLTextPanel:
    """Small self-contained text overlay for numerical adapter windows."""

    def __init__(self) -> None:
        import pygame
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_CLAMP_TO_EDGE, GL_DYNAMIC_DRAW, GL_FALSE,
            GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINEAR, GL_RGBA, GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
            glBindBuffer, glBindTexture, glBindVertexArray, glBufferData,
            glEnableVertexAttribArray, glGenBuffers, glGenTextures,
            glGenVertexArrays, glTexImage2D, glTexParameteri,
            glVertexAttribPointer,
        )
        from OpenGL.GL.shaders import compileProgram, compileShader

        self._gl = __import__("OpenGL.GL", fromlist=("*",))
        self._program = compileProgram(
            compileShader(
                """#version 330 core
                layout(location=0) in vec2 pos;
                layout(location=1) in vec2 uv_in;
                out vec2 uv;
                void main(){ uv=uv_in; gl_Position=vec4(pos,0,1); }""",
                GL_VERTEX_SHADER,
            ),
            compileShader(
                """#version 330 core
                in vec2 uv; out vec4 color; uniform sampler2D panel;
                void main(){ color=texture(panel,uv); }""",
                GL_FRAGMENT_SHADER,
            ),
        )
        self._vao = int(glGenVertexArrays(1))
        self._vbo = int(glGenBuffers(1))
        self._texture = int(glGenTextures(1))
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, 64, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindTexture(GL_TEXTURE_2D, self._texture)
        for name in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            glTexParameteri(GL_TEXTURE_2D, name, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        pygame.font.init()
        self._font = pygame.font.SysFont("consolas,monospace", 15)
        self._surface = None
        self._texture_size = (1, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, bytes((0, 0, 0, 0)),
        )

    def update(self, lines: Sequence[str]) -> None:
        import pygame
        from OpenGL.GL import (
            GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture,
            glTexImage2D,
        )

        rendered = [
            self._font.render(line or " ", True, (210, 228, 245))
            for line in lines
        ]
        width = max((line.get_width() for line in rendered), default=1) + 28
        height = sum(line.get_height() for line in rendered) + 24
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((7, 12, 24, 224))
        y = 12
        for line in rendered:
            surface.blit(line, (14, y))
            y += line.get_height()
        raw = pygame.image.tobytes(surface, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, self._texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
            GL_UNSIGNED_BYTE, raw,
        )
        self._texture_size = (width, height)

    def draw(self, window_size: tuple[int, int]) -> None:
        gl = self._gl
        width, height = window_size
        panel_width = min(self._texture_size[0], max(1, width - 20))
        panel_height = min(self._texture_size[1], max(1, height - 20))
        left = 1.0 - 2.0 * (panel_width + 12) / width
        right = 1.0 - 24.0 / width
        top = 1.0 - 24.0 / height
        bottom = top - 2.0 * panel_height / height
        vertices = np.asarray((
            (left, bottom, 0, 0), (right, bottom, 1, 0),
            (left, top, 0, 1), (right, top, 1, 1),
        ), dtype=np.float32)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self._program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW
        )
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)

    def close(self) -> None:
        gl = self._gl
        gl.glDeleteTextures(1, (self._texture,))
        gl.glDeleteBuffers(1, (self._vbo,))
        gl.glDeleteVertexArrays(1, (self._vao,))
        gl.glDeleteProgram(self._program)


def _heatmap_rgba(
    values: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    """Color a scalar field with Pluck's blue/cyan/gold/red instrument palette."""
    field = np.asarray(values, dtype=np.float32)
    scale = max(float(upper) - float(lower), np.finfo(np.float32).eps)
    unit = np.clip((field - float(lower)) / scale, 0.0, 1.0)
    stops = np.asarray(
        (
            (0.015, 0.025, 0.12),
            (0.06, 0.34, 0.88),
            (0.02, 0.88, 0.88),
            (0.98, 0.78, 0.12),
            (0.94, 0.12, 0.045),
        ),
        dtype=np.float32,
    )
    position = unit * (len(stops) - 1)
    index = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = (position - index)[..., None]
    rgb = stops[index] * (1.0 - fraction) + stops[index + 1] * fraction
    alpha = np.ones((*field.shape, 1), dtype=np.float32)
    return np.asarray(np.concatenate((rgb, alpha), axis=-1) * 255, np.uint8)


class HeatmapDashboard:
    """Persistent Pluck-style OpenGL dashboard for three fields and one trace.

    Each panel is a stateful texture.  Repeated updates use ``glTexSubImage2D``
    when dimensions are unchanged, avoiding texture-object churn.
    """

    def __init__(
        self,
        *,
        title: str = "Pluck field dashboard",
        size: tuple[int, int] = (1280, 800),
    ) -> None:
        import pygame
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_CLAMP_TO_EDGE, GL_DYNAMIC_DRAW, GL_FALSE,
            GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINEAR, GL_RGBA, GL_RGBA8,
            GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE,
            GL_VERTEX_SHADER, glBindBuffer, glBindTexture, glBindVertexArray,
            glBufferData, glEnableVertexAttribArray, glGenBuffers,
            glGenTextures, glGenVertexArrays, glTexImage2D, glTexParameteri,
            glVertexAttribPointer,
        )
        from OpenGL.GL.shaders import compileProgram, compileShader

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode(
            size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption(title)
        pygame.font.init()
        self._pygame = pygame
        self._gl = __import__("OpenGL.GL", fromlist=("*",))
        self._font = pygame.font.SysFont("consolas,monospace", 17)
        self._small_font = pygame.font.SysFont("consolas,monospace", 13)
        self._program = compileProgram(
            compileShader(
                """#version 330 core
                layout(location=0) in vec2 pos;
                layout(location=1) in vec2 uv_in;
                out vec2 uv;
                void main(){ uv=uv_in; gl_Position=vec4(pos,0,1); }""",
                GL_VERTEX_SHADER,
            ),
            compileShader(
                """#version 330 core
                in vec2 uv; out vec4 color; uniform sampler2D panel;
                void main(){ color=texture(panel,uv); }""",
                GL_FRAGMENT_SHADER,
            ),
        )
        self._vao = int(glGenVertexArrays(1))
        self._vbo = int(glGenBuffers(1))
        self._textures = tuple(int(value) for value in glGenTextures(4))
        self._texture_sizes = [(0, 0)] * 4
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, 64, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        for texture in self._textures:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, bytes((4, 7, 15, 255)),
            )
        self._closed = False
        self._clock = pygame.time.Clock()

    def _captioned_surface(
        self, rgba: np.ndarray, title: str, subtitle: str = ""
    ):
        pygame = self._pygame
        panel_width, field_height = 640, 360
        band = 52
        surface = pygame.Surface(
            (panel_width, field_height + band), pygame.SRCALPHA
        )
        pixels = pygame.surfarray.make_surface(
            np.ascontiguousarray(rgba[..., :3].swapaxes(0, 1))
        )
        pixels = pygame.transform.smoothscale(
            pixels, (panel_width, field_height)
        )
        surface.blit(pixels, (0, band))
        surface.fill((5, 10, 22, 255), (0, 0, panel_width, band))
        surface.blit(self._font.render(title, True, (226, 238, 250)), (12, 6))
        if subtitle:
            surface.blit(
                self._small_font.render(subtitle, True, (128, 172, 206)),
                (12, 30),
            )
        return surface

    def _loss_surface(
        self,
        losses: Sequence[float],
        size: tuple[int, int],
        status_lines: Sequence[str],
    ):
        pygame = self._pygame
        width, height = size
        surface = pygame.Surface(size, pygame.SRCALPHA)
        surface.fill((5, 10, 22, 255))
        surface.blit(
            self._font.render("training loss", True, (226, 238, 250)), (14, 8)
        )
        values = np.asarray(losses, dtype=np.float64)
        plot = pygame.Rect(54, 52, max(20, width - 72), max(30, height - 132))
        pygame.draw.rect(surface, (11, 21, 38), plot)
        for index in range(5):
            y = plot.top + index * plot.height // 4
            pygame.draw.line(
                surface, (31, 52, 72), (plot.left, y), (plot.right, y), 1
            )
        finite = values[np.isfinite(values) & (values > 0)]
        if len(finite):
            logs = np.log10(np.maximum(values, np.finfo(np.float64).tiny))
            lower, upper = float(logs.min()), float(logs.max())
            if upper - lower < 1e-9:
                upper = lower + 1.0
            xs = np.linspace(plot.left, plot.right, len(logs))
            ys = plot.bottom - (logs - lower) / (upper - lower) * plot.height
            points = [(int(x), int(y)) for x, y in zip(xs, ys)]
            if len(points) > 1:
                pygame.draw.lines(surface, (248, 99, 38), False, points, 3)
            elif points:
                pygame.draw.circle(surface, (248, 99, 38), points[0], 3)
            summary = f"{values[0]:.4g} -> {values[-1]:.4g}   n={len(values)}"
            surface.blit(
                self._small_font.render(summary, True, (244, 169, 111)),
                (14, height - 70),
            )
        for index, line in enumerate(status_lines[:2]):
            surface.blit(
                self._small_font.render(str(line), True, (128, 172, 206)),
                (14, height - 48 + index * 18),
            )
        return surface

    def _upload_surface(self, index: int, surface) -> None:
        gl = self._gl
        width, height = surface.get_size()
        raw = self._pygame.image.tobytes(surface, "RGBA", True)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._textures[index])
        if self._texture_sizes[index] == (width, height):
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, width, height,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, raw,
            )
        else:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, raw,
            )
            self._texture_sizes[index] = (width, height)

    def update(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        error: np.ndarray,
        losses: Sequence[float],
        *,
        status_lines: Sequence[str] = (),
    ) -> None:
        error_limit = max(0.25, float(np.nanmax(error, initial=0.0)))
        panels = (
            self._captioned_surface(
                _heatmap_rgba(target, -1.0, 1.0),
                "continuous target",
                "independent analytic field",
            ),
            self._captioned_surface(
                _heatmap_rgba(prediction, -1.0, 1.0),
                "FusedProgram network",
                "AbstractNN prediction",
            ),
            self._captioned_surface(
                _heatmap_rgba(error, 0.0, error_limit),
                "absolute error",
                f"range 0 .. {error_limit:.4g}",
            ),
        )
        for index, panel in enumerate(panels):
            self._upload_surface(index, panel)
        loss_size = panels[0].get_size()
        self._upload_surface(
            3, self._loss_surface(losses, loss_size, status_lines)
        )

    def _draw_texture(self, texture: int, rect: tuple[float, float, float, float]):
        gl = self._gl
        left, bottom, right, top = rect
        vertices = np.asarray(
            (
                (left, bottom, 0, 0), (right, bottom, 1, 0),
                (left, top, 0, 1), (right, top, 1, 1),
            ),
            dtype=np.float32,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW
        )
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    def render(self) -> None:
        gl = self._gl
        width, height = self._pygame.display.get_surface().get_size()
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.012, 0.018, 0.03, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self._program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self._vao)
        margin = 0.025
        gap = 0.018
        half_w = (2.0 - 2 * margin - gap) * 0.5
        half_h = (2.0 - 2 * margin - gap) * 0.5
        x0, x1 = -1.0 + margin, -1.0 + margin + half_w
        x2, x3 = x1 + gap, 1.0 - margin
        y0, y1 = -1.0 + margin, -1.0 + margin + half_h
        y2, y3 = y1 + gap, 1.0 - margin
        for texture, rect in zip(
            self._textures,
            ((x0, y2, x1, y3), (x2, y2, x3, y3),
             (x0, y0, x1, y1), (x2, y0, x3, y1)),
        ):
            self._draw_texture(texture, rect)
        gl.glBindVertexArray(0)

    def pump(self) -> bool:
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            if event.type == self._pygame.KEYDOWN and event.key == self._pygame.K_ESCAPE:
                return False
        self.render()
        self._pygame.display.flip()
        return True

    def save(self, path: str | Path) -> Path:
        from PIL import Image

        self.render()
        width, height = self._pygame.display.get_surface().get_size()
        raw = self._gl.glReadPixels(
            0, 0, width, height, self._gl.GL_RGBA, self._gl.GL_UNSIGNED_BYTE
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGBA", (width, height), raw).transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        ).convert("RGB").save(output)
        return output.resolve()

    def wait(self) -> None:
        while self.pump():
            self._clock.tick(60)

    def close(self) -> None:
        if self._closed:
            return
        gl = self._gl
        gl.glDeleteTextures(len(self._textures), self._textures)
        gl.glDeleteBuffers(1, (self._vbo,))
        gl.glDeleteVertexArrays(1, (self._vao,))
        gl.glDeleteProgram(self._program)
        self._pygame.quit()
        self._closed = True


def view_triangle_mesh(
    triangles: np.ndarray,
    *,
    title: str = "Pluck ordinary OpenGL mesh viewer",
    size: tuple[int, int] = (1000, 760),
    max_frames: int | None = None,
    triangle_values: np.ndarray | None = None,
    value_label: str = "scalar value",
) -> None:
    """Open an orbiting view, optionally colored by signed triangle values."""
    import pygame
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        glClear,
        glClearColor,
        glDeleteBuffers,
        glDeleteVertexArrays,
        glEnable,
        glViewport,
    )

    from base_gl_renderer import BaseGLRenderer
    from material_db import MaterialDatabase

    rows = triangle_mesh_vertex_rows(triangles)
    if not len(rows):
        raise ValueError("cannot display an empty triangle mesh")

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption(title)

    database = MaterialDatabase()
    if triangle_values is None:
        material_ids = database.register(
            "youngman_surface",
            {
                "albedo_rgb": [0.18, 0.56, 0.92],
                "roughness": 0.32,
                "metallic": 0.08,
                "ior": 1.48,
                "opacity": 1.0,
                "emission_rgb": [0.035, 0.10, 0.18],
            },
        )
    else:
        triangle_values = np.asarray(triangle_values, dtype=np.float64)
        triangle_count = rows.shape[0] // 3
        if triangle_values.shape != (triangle_count,):
            raise ValueError("triangle_values needs one scalar per triangle")
        palette_bins, value_limit = scalar_triangle_bins(triangle_values)
        palette_ids = []
        for index in range(33):
            position = 2.0 * index / 32.0 - 1.0
            color = _diverging_color(position)
            palette_ids.append(database.register(
                f"scalar_{index:02d}",
                {
                    "albedo_rgb": color,
                    "roughness": 0.38,
                    "metallic": 0.04,
                    "ior": 1.46,
                    "opacity": 1.0,
                    "emission_rgb": (0.12 * np.asarray(color)).tolist(),
                },
            ))
        material_ids = np.repeat(
            np.asarray(palette_ids, dtype=np.int32)[palette_bins], 3
        )
        pygame.display.set_caption(
            f"{title} | {value_label}: blue -{value_limit:.3g}, red +{value_limit:.3g}"
        )
    renderer = BaseGLRenderer(database, auto_drain=False)
    renderer.init_gl()
    renderer.set_point_lights(
        np.asarray(((3.5, -4.0, 4.5),), np.float32),
        np.asarray(((1.0, 0.92, 0.82),), np.float32),
        np.asarray((16.0,), np.float32),
    )
    resources = _upload_mesh(rows, material_ids)
    vao, buffers = resources[0], resources[1:]

    points = rows[:, :3]
    center = points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1))) or 1.0
    clock = pygame.time.Clock()
    angle = 0.0
    frame = 0
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            width, height = pygame.display.get_surface().get_size()
            glViewport(0, 0, width, height)
            glEnable(GL_DEPTH_TEST)
            glClearColor(0.012, 0.018, 0.03, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            angle += clock.get_time() * 0.00022
            eye = center + radius * np.asarray(
                (2.8 * math.cos(angle), 2.8 * math.sin(angle), 1.65),
                dtype=np.float32,
            )
            view = _look_at(eye, center)
            projection = _perspective(
                math.radians(46.0), width / max(1, height), radius * 0.02, radius * 12
            )
            renderer.draw_mesh(vao, len(rows), projection @ view, view)
            pygame.display.flip()
            clock.tick(60)
            frame += 1
            if max_frames is not None and frame >= max_frames:
                running = False
    finally:
        glDeleteBuffers(len(buffers), buffers)
        glDeleteVertexArrays(1, (vao,))
        pygame.quit()


def view_profiled_triangle_mesh_stream(
    solve_frame,
    *,
    title: str = "Pluck profiled geometry stream",
    size: tuple[int, int] = (1280, 800),
    period_sec: float = 8.0,
    max_solves: int | None = None,
    max_frames: int | None = None,
) -> dict:
    """Display latest-result-wins solved meshes while solving off the GL thread.

    ``solve_frame(index, time_value)`` returns :class:`LiveMeshFrame`. The
    worker never calls OpenGL. If solving outruns display, stale unpublished
    frames are dropped rather than allowing an unbounded queue.
    """
    import time
    import pygame
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, glClear,
        glClearColor, glDeleteBuffers, glDeleteVertexArrays, glEnable,
        glViewport,
    )
    from base_gl_renderer import BaseGLRenderer
    from material_db import MaterialDatabase

    updates: Queue = Queue(maxsize=1)
    stopped = Event()

    def publish(frame):
        try:
            updates.put_nowait(frame)
        except Full:
            try:
                updates.get_nowait()
            except Empty:
                pass
            updates.put_nowait(frame)

    def worker():
        index = 0
        started = time.perf_counter()
        try:
            while not stopped.is_set() and (
                max_solves is None or index < max_solves
            ):
                time_value = (
                    (time.perf_counter() - started) / max(period_sec, 1e-9)
                ) % 1.0
                publish(solve_frame(index, time_value))
                index += 1
        except BaseException as error:
            publish(error)

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption(title)
    database = MaterialDatabase()
    palette_ids = []
    geometry_material_id = database.register(
        "live_geometry",
        {
            "albedo_rgb": [0.16, 0.52, 0.88],
            "roughness": 0.3,
            "metallic": 0.12,
            "ior": 1.48,
            "opacity": 1.0,
            "emission_rgb": [0.02, 0.07, 0.13],
        },
    )
    for index in range(33):
        position = 2.0 * index / 32.0 - 1.0
        color = _diverging_color(position)
        palette_ids.append(database.register(
            f"live_scalar_{index:02d}",
            {
                "albedo_rgb": color,
                "roughness": 0.38,
                "metallic": 0.04,
                "ior": 1.46,
                "opacity": 1.0,
                "emission_rgb": (0.12 * np.asarray(color)).tolist(),
            },
        ))
    renderer = BaseGLRenderer(database, auto_drain=False)
    renderer.init_gl()
    renderer.set_point_lights(
        np.asarray(((3.5, -4.0, 4.5),), np.float32),
        np.asarray(((1.0, 0.92, 0.82),), np.float32),
        np.asarray((16.0,), np.float32),
    )
    panel = _OpenGLTextPanel()
    panel.update(("waiting for first complete solve...",))
    resources = None
    rows = None
    center = np.zeros(3, dtype=np.float32)
    radius = 1.0
    thread = Thread(target=worker, name="profiled-mesh-solver", daemon=True)
    thread.start()
    clock = pygame.time.Clock()
    angle = 0.0
    frame_count = 0
    running = True
    session_started = perf_counter()
    video_history: dict[str, list[float]] = {
        name: [] for name in (
            "mesh_upload", "draw_submit", "hud", "swap", "frame", "publish_latency"
        )
    }
    base_panel_lines = ("waiting for first complete solve...",)
    last_panel_refresh = 0.0

    def record_video(name: str, elapsed: float) -> None:
        values = video_history[name]
        values.append(float(elapsed))
        if len(values) > 600:
            del values[:-600]

    def video_lines() -> tuple[str, ...]:
        lines = ["", "video CPU wall profile", "stage                  mean       p95"]
        for name, values in video_history.items():
            if not values:
                continue
            array = np.asarray(values)
            lines.append(
                f"{name:<20} {array.mean()*1e3:7.2f} "
                f"{np.quantile(array, .95)*1e3:8.2f} ms"
            )
        lines.extend((
            f"rendered frames      {frame_count:8d}",
            f"session total       {perf_counter()-session_started:8.2f} s",
            "GPU completion is not forced",
        ))
        return tuple(lines)
    try:
        while running:
            frame_started = perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            latest = None
            while True:
                try:
                    latest = updates.get_nowait()
                except Empty:
                    break
            if latest is not None:
                if isinstance(latest, BaseException):
                    raise latest
                upload_started = perf_counter()
                new_rows = triangle_mesh_vertex_rows(latest.triangles)
                if latest.triangle_values is None:
                    limit = None
                    material_ids = np.full(
                        len(new_rows), geometry_material_id, dtype=np.int32
                    )
                else:
                    bins, limit = scalar_triangle_bins(latest.triangle_values)
                    material_ids = np.repeat(
                        np.asarray(palette_ids, dtype=np.int32)[bins], 3
                    )
                new_resources = _upload_mesh(new_rows, material_ids)
                if resources is not None:
                    glDeleteBuffers(len(resources) - 1, resources[1:])
                    glDeleteVertexArrays(1, (resources[0],))
                resources, rows = new_resources, new_rows
                points = rows[:, :3]
                center = points.mean(axis=0)
                radius = (
                    float(np.max(np.linalg.norm(points - center, axis=1))) or 1.0
                )
                record_video("mesh_upload", perf_counter() - upload_started)
                record_video(
                    "publish_latency", perf_counter() - latest.published_at
                )
                scale_line = (
                    "display            geometry material"
                    if limit is None
                    else f"color scale       +/- {limit:.5g}"
                )
                base_panel_lines = tuple(latest.panel_lines) + (
                    "", scale_line, "ESC closes; newest complete solve wins",
                )
            width, height = pygame.display.get_surface().get_size()
            glViewport(0, 0, width, height)
            glEnable(GL_DEPTH_TEST)
            glClearColor(0.012, 0.018, 0.03, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if resources is not None and rows is not None:
                draw_started = perf_counter()
                angle += clock.get_time() * 0.00022
                eye = center + radius * np.asarray(
                    (2.8 * math.cos(angle), 2.8 * math.sin(angle), 1.65),
                    dtype=np.float32,
                )
                view = _look_at(eye, center)
                projection = _perspective(
                    math.radians(46.0), width / max(1, height),
                    radius * 0.02, radius * 12,
                )
                renderer.draw_mesh(
                    resources[0], len(rows), projection @ view, view
                )
                record_video("draw_submit", perf_counter() - draw_started)
            now = perf_counter()
            if now - last_panel_refresh >= 0.5:
                panel.update(base_panel_lines + video_lines())
                last_panel_refresh = now
            hud_started = perf_counter()
            panel.draw((width, height))
            record_video("hud", perf_counter() - hud_started)
            swap_started = perf_counter()
            pygame.display.flip()
            record_video("swap", perf_counter() - swap_started)
            clock.tick(60)
            frame_count += 1
            record_video("frame", perf_counter() - frame_started)
            if max_frames is not None and frame_count >= max_frames:
                running = False
    finally:
        stopped.set()
        thread.join(timeout=2.0)
        if resources is not None:
            glDeleteBuffers(len(resources) - 1, resources[1:])
            glDeleteVertexArrays(1, (resources[0],))
        panel.close()
        pygame.quit()
    summary = summarize_video_profile(
        video_history,
        rendered_frames=frame_count,
        session_elapsed_sec=perf_counter() - session_started,
    )
    print(
        f"[video stats] total={summary['session_elapsed_sec']:.3f}s "
        f"frames={summary['rendered_frames']} gpu_finish=false",
        flush=True,
    )
    for name, values in summary["stages"].items():
        print(
            f"  {name:<20} mean={values['mean_sec']*1e3:8.3f}ms "
            f"p95={values['p95_sec']*1e3:8.3f}ms "
            f"max={values['max_sec']*1e3:8.3f}ms n={values['count']}",
            flush=True,
        )
    return summary
