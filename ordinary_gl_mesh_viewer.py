"""Small public triangle-mesh viewer for Pluck's ordinary OpenGL renderer.

This adapter owns window, VAO, and camera mechanics while delegating material
shading to :class:`base_gl_renderer.BaseGLRenderer`.  Numerical projects can
therefore present a triangle soup in Pluck without importing the full game.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Tuple

import numpy as np


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
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
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
