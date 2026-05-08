"""GLSL ownership and compilation skeleton lifted out of demo_pluck_gl.py.

This module is the intended home for shader source strings and program
construction that currently live inline inside demo_pluck_gl.py.

Associated repository files
---------------------------
Current shader ownership and call sites:
- demo_pluck_gl.py
- camera_designer_station.py
- opengl_widget.py
- _archive/_extract_shader.py
- _archive/demo_pluck_gl.old.py

Existing standalone shader assets:
- csrc/shaders/coherent_accumulate.comp.glsl

Scene, material, and profile sources consumed by the compute path:
- material_db.py
- spectral_material.py
- ray_tracer_bridge.py
- camera_designer/scene_builder.py
- guitar_part.py

Camera and sensor systems that need the compiled programs:
- camera_item.py
- camera_panel.py
- camera_software/base.py
- camera_software/lens_manifold.py
- camera_software/camera_back.py

Intended end state
------------------
1. Shader source strings live here instead of inside demo_pluck_gl.py.
2. Program compilation and caching live here instead of using demo-local
   helper functions.
3. Demo and non-demo OpenGL callers import compiled programs from one place.
4. The unified transport bridge can ask this module for GLSL backend resources
   without importing the demo application module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ShaderSourceSet:
    march_vs: str = ""
    march_fs: str = ""
    gpu_ray_field_cs: str = ""
    gpu_sensor_cs: str = ""
    sensor_blit_vs: str = ""
    sensor_blit_fs: str = ""
    decay_cs: str = ""
    extra_sources: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ShaderProgramSet:
    program_ids: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_source_set() -> ShaderSourceSet:
    """Return the canonical GLSL source bundle for the ray-tracing stack."""
    raise NotImplementedError


def compile_shader_source(source: str, shader_kind: int) -> int:
    """Compile a single GLSL shader object in the current GL context."""
    raise NotImplementedError


def link_program(*shader_ids: int) -> int:
    """Link one GLSL program from precompiled shader object IDs."""
    raise NotImplementedError


def compile_program_set(source_set: ShaderSourceSet | None = None) -> ShaderProgramSet:
    """Compile the canonical ray-tracing GLSL program suite."""
    raise NotImplementedError


def install_demo_sources() -> ShaderSourceSet:
    """Temporary migration hook for moving shader strings out of demo_pluck_gl.py."""
    raise NotImplementedError
