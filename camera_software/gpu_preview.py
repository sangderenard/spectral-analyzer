"""Nonblocking shared-OpenGL preview products and capture.

Producers own textures and publish immutable descriptions. Hosts own tabs,
presentation, and capture policy. No producer performs presentation I/O.

An embedding program creates its display context, passes
``OpenGLShareGroup.current()`` to the background camera/solver service, and
gives both sides one ``PreviewProductRegistry``. Camera pipelines may use
``RayPipelinePreviewBridge`` directly. Other solvers use
``PreviewProductPublisher.publish_accumulation()``,
``publish_processing_group()``, or ``publish_complex_field()``. This keeps the
asset-request service independent from whichever UI presents its products.
"""
from __future__ import annotations

import ctypes
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar


_RequestT = TypeVar("_RequestT")
_ProductT = TypeVar("_ProductT")


class PreviewProductKind(str, Enum):
    IMAGE = "image"
    ACCUMULATION = "accumulation"
    PROCESSING_GROUP = "processing-group"
    DEPTH = "depth"
    COMPLEX_FIELD = "complex-field"
    CAMERA_GEOMETRY = "camera-geometry"
    LIGHT_FIELD = "light-field"


@dataclass(frozen=True, slots=True)
class PreviewTextureProduct:
    product_id: str
    tab_label: str
    texture_id: int
    width: int
    height: int
    generation: int
    producer: str
    depth: int = 1
    texture_target: int = 0x0DE1  # GL_TEXTURE_2D
    internal_format: int = 0x8814  # GL_RGBA32F
    kind: PreviewProductKind = PreviewProductKind.IMAGE
    color_space: str = "linear-srgb"
    alpha_mode: str = "straight"
    orientation: str = "bottom-left"
    sync_handle: Any = None
    group_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    published_at_s: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not str(self.product_id).strip():
            raise ValueError("preview product_id must be non-empty")
        if not str(self.tab_label).strip():
            raise ValueError("preview tab_label must be non-empty")
        if int(self.texture_id) <= 0:
            raise ValueError("preview texture_id must be positive")
        if int(self.width) <= 0 or int(self.height) <= 0 or int(self.depth) <= 0:
            raise ValueError("preview dimensions must be positive")
        if int(self.generation) < 0:
            raise ValueError("preview generation cannot be negative")
        if self.orientation not in {"bottom-left", "top-left"}:
            raise ValueError("preview orientation must be bottom-left or top-left")
        if self.alpha_mode not in {"straight", "premultiplied", "opaque"}:
            raise ValueError("unsupported preview alpha mode")


class PreviewProductRegistry:
    """Atomic latest-generation registry shared by producers and a GL host."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._products: dict[str, PreviewTextureProduct] = {}
        self._order: list[str] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def publish(self, product: PreviewTextureProduct) -> bool:
        """Publish a newer generation; stale/duplicate generations are ignored."""
        key = str(product.product_id)
        with self._lock:
            current = self._products.get(key)
            if current is not None and int(product.generation) <= int(current.generation):
                return False
            self._products[key] = product
            if key not in self._order:
                self._order.append(key)
            self._revision += 1
            return True

    def withdraw(self, product_id: str) -> bool:
        key = str(product_id)
        with self._lock:
            if key not in self._products:
                return False
            del self._products[key]
            self._order = [item for item in self._order if item != key]
            self._revision += 1
            return True

    def get(self, product_id: str) -> PreviewTextureProduct | None:
        with self._lock:
            return self._products.get(str(product_id))

    def snapshot(self) -> tuple[PreviewTextureProduct, ...]:
        with self._lock:
            return tuple(
                self._products[key] for key in self._order
                if key in self._products
            )

    def products_for_group(self, group_id: str) -> tuple[PreviewTextureProduct, ...]:
        wanted = str(group_id)
        return tuple(item for item in self.snapshot() if item.group_id == wanted)


@dataclass(frozen=True, slots=True)
class LatestProductionResult(Generic[_ProductT]):
    """One completed asynchronous production attempt."""

    request_id: int
    payload: _ProductT | None
    started_at_s: float
    finished_at_s: float
    error: BaseException | None = None

    @property
    def elapsed_s(self) -> float:
        return max(0.0, float(self.finished_at_s) - float(self.started_at_s))


class LatestOnlyProducer(Generic[_RequestT, _ProductT]):
    """Run expensive production off-thread with a one-item latest-wins inbox.

    UI/control code may request freely without building an unbounded queue of
    obsolete frames. The worker completes its current request, then consumes
    only the newest pending state. Polling never waits.
    """

    def __init__(
        self,
        produce: Callable[[_RequestT], _ProductT],
        *,
        name: str = "LatestOnlyProducer",
    ) -> None:
        self._produce = produce
        self._condition = threading.Condition()
        self._pending: tuple[int, _RequestT] | None = None
        self._latest: LatestProductionResult[_ProductT] | None = None
        self._next_request_id = 0
        self._busy = False
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name=str(name), daemon=True
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._busy

    def request(self, value: _RequestT) -> int:
        with self._condition:
            if self._stopping:
                raise RuntimeError("latest-only producer is closed")
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending = (request_id, value)
            self._condition.notify()
            return request_id

    def poll(
        self, *, after_request_id: int = -1
    ) -> LatestProductionResult[_ProductT] | None:
        with self._condition:
            if (
                self._latest is None
                or self._latest.request_id <= int(after_request_id)
            ):
                return None
            return self._latest

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    self._busy = False
                    self._condition.notify_all()
                    return
                request_id, value = self._pending
                self._pending = None
                self._busy = True
            started = time.monotonic()
            payload: _ProductT | None = None
            error: BaseException | None = None
            try:
                payload = self._produce(value)
            except BaseException as exc:
                error = exc
            finished = time.monotonic()
            with self._condition:
                self._latest = LatestProductionResult(
                    request_id=request_id,
                    payload=payload,
                    started_at_s=started,
                    finished_at_s=finished,
                    error=error,
                )
                self._busy = False
                self._condition.notify_all()

    def close(self, *, timeout_s: float = 5.0) -> bool:
        """Stop accepting work and wait briefly for current production."""

        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout_s)))
        return not self._thread.is_alive()


@dataclass(frozen=True, slots=True)
class OpenGLShareGroup:
    """Native handles a background compute context uses for object sharing."""

    context_handle: int
    device_context_handle: int

    @classmethod
    def current(cls) -> "OpenGLShareGroup":
        if os.name != "nt":
            return cls(0, 0)
        opengl32 = ctypes.windll.opengl32
        opengl32.wglGetCurrentContext.restype = ctypes.c_void_p
        opengl32.wglGetCurrentDC.restype = ctypes.c_void_p
        return cls(
            int(opengl32.wglGetCurrentContext() or 0),
            int(opengl32.wglGetCurrentDC() or 0),
        )

    def configure_tracer(self, tracer: Any) -> None:
        """Configure the existing native ray pipeline before its first submit."""
        if self.context_handle <= 0 or self.device_context_handle <= 0:
            raise RuntimeError("no current Windows OpenGL context/share group")
        tracer.set_gl_display_hglrc(self.context_handle)
        tracer.set_gl_display_hdc(self.device_context_handle)


class PreviewProductPublisher:
    """Small producer-side adapter for camera, pipeline, and wave textures."""

    def __init__(
        self, registry: PreviewProductRegistry, producer: str, *, group_id: str = "",
    ) -> None:
        self.registry = registry
        self.producer = str(producer)
        self.group_id = str(group_id)
        self._generations: dict[str, int] = {}

    def publish(
        self,
        product_id: str,
        tab_label: str,
        texture_id: int,
        width: int,
        height: int,
        *,
        depth: int = 1,
        kind: PreviewProductKind = PreviewProductKind.IMAGE,
        generation: int | None = None,
        sync_handle: Any = None,
        texture_target: int = 0x0DE1,
        internal_format: int = 0x8814,
        color_space: str = "linear-srgb",
        alpha_mode: str = "straight",
        orientation: str = "bottom-left",
        metadata: Mapping[str, Any] | None = None,
    ) -> PreviewTextureProduct:
        key = str(product_id)
        if generation is None:
            generation = self._generations.get(key, 0) + 1
        self._generations[key] = max(
            int(generation), self._generations.get(key, 0)
        )
        product = PreviewTextureProduct(
            product_id=key,
            tab_label=str(tab_label),
            texture_id=int(texture_id),
            width=int(width),
            height=int(height),
            depth=int(depth),
            generation=int(generation),
            producer=self.producer,
            texture_target=int(texture_target),
            internal_format=int(internal_format),
            kind=kind,
            color_space=str(color_space),
            alpha_mode=str(alpha_mode),
            orientation=str(orientation),
            sync_handle=sync_handle,
            group_id=self.group_id,
            metadata=dict(metadata or {}),
        )
        self.registry.publish(product)
        return product

    def publish_accumulation(self, *args: Any, **kwargs: Any) -> PreviewTextureProduct:
        kwargs["kind"] = PreviewProductKind.ACCUMULATION
        return self.publish(*args, **kwargs)

    def publish_processing_group(self, *args: Any, **kwargs: Any) -> PreviewTextureProduct:
        kwargs["kind"] = PreviewProductKind.PROCESSING_GROUP
        return self.publish(*args, **kwargs)

    def publish_complex_field(self, *args: Any, **kwargs: Any) -> PreviewTextureProduct:
        kwargs["kind"] = PreviewProductKind.COMPLEX_FIELD
        return self.publish(*args, **kwargs)

    def publish_camera_geometry(self, *args: Any, **kwargs: Any) -> PreviewTextureProduct:
        kwargs["kind"] = PreviewProductKind.CAMERA_GEOMETRY
        return self.publish(*args, **kwargs)

    def publish_light_field(self, *args: Any, **kwargs: Any) -> PreviewTextureProduct:
        kwargs["kind"] = PreviewProductKind.LIGHT_FIELD
        return self.publish(*args, **kwargs)

    def publish_texture_info(
        self, product_id: str, tab_label: str, info: Mapping[str, Any], **kwargs: Any,
    ) -> PreviewTextureProduct | None:
        if not info or int(info.get("texture_id", 0)) <= 0:
            return None
        return self.publish(
            product_id,
            tab_label,
            int(info["texture_id"]),
            int(info["width"]),
            int(info["height"]),
            depth=int(info.get("depth", 1)),
            generation=int(info["generation"]),
            texture_target=int(info.get("texture_target", 0x0DE1)),
            internal_format=int(info.get("internal_format", 0x8814)),
            **kwargs,
        )


class RayPipelinePreviewBridge:
    """Publish existing native camera/pipeline textures into preview tabs."""

    def __init__(self, registry: PreviewProductRegistry, tracer: Any) -> None:
        self.tracer = tracer
        self.camera = PreviewProductPublisher(
            registry, "spectral-camera", group_id="camera-pipeline"
        )
        self.field = PreviewProductPublisher(
            registry, "optical-complex-transport", group_id="camera-pipeline"
        )
        self.wave = PreviewProductPublisher(
            registry, "wave-arena", group_id="camera-pipeline"
        )

    def poll(self) -> tuple[PreviewTextureProduct, ...]:
        """Publish newly completed native generations without pixel readback."""
        published: list[PreviewTextureProduct] = []
        surface_info = getattr(self.tracer, "surface_scan_texture_info", None)
        if callable(surface_info):
            item = self.camera.publish_texture_info(
                "camera.surface-scan", "SURFACE SCAN", surface_info(),
                kind=PreviewProductKind.IMAGE,
                metadata={"stage": "surface-scan", "capture": "scientific-rgba"},
            )
            if item is not None:
                published.append(item)
        field_info = getattr(self.tracer, "get_field_display_texture_info", None)
        if callable(field_info):
            item = self.field.publish_texture_info(
                "transport.complex-accumulation", "COMPLEX TRANSPORT", field_info(),
                kind=PreviewProductKind.COMPLEX_FIELD,
                metadata={
                    "stage": "complex-hit-accumulation",
                    "view": "orthographic-slice",
                    "slice": 0.5,
                    "representation": "ray-carried-complex-amplitude",
                    "wave_solver": False,
                },
            )
            if item is not None:
                published.append(item)
        wave_info = getattr(self.tracer, "get_wave_arena_texture_info", None)
        if callable(wave_info):
            info = wave_info()
            item = self.wave.publish_texture_info(
                "wave.arena-state", "WAVE ARENA", info,
                kind=PreviewProductKind.COMPLEX_FIELD,
                metadata={
                    "stage": "t4-angular-spectrum",
                    "view": "physical-exit-plane",
                    "encoding": "phase-hue-amplitude-value",
                    "representation": "transverse-complex-field",
                    "wave_solver": True,
                    "arena_id": int(info.get("arena_id", -1)) if info else -1,
                    "band": int(info.get("band", -1)) if info else -1,
                    "direction": int(info.get("direction", 0)) if info else 0,
                    "capture": "presentation-rgba",
                },
            )
            if item is not None:
                published.append(item)
        geometry_info = getattr(
            self.tracer, "get_camera_geometry_texture_info", None
        )
        if callable(geometry_info):
            item = self.camera.publish_texture_info(
                "camera.geometry", "CAMERA GEOMETRY", geometry_info(),
                kind=PreviewProductKind.CAMERA_GEOMETRY,
                metadata={"stage": "camera-geometry", "units": "metres"},
            )
            if item is not None:
                published.append(item)
        light_field_info = getattr(
            self.tracer, "get_light_field_texture_info", None
        )
        if callable(light_field_info):
            item = self.camera.publish_texture_info(
                "camera.light-field", "LIGHT FIELD", light_field_info(),
                kind=PreviewProductKind.LIGHT_FIELD,
                metadata={"stage": "light-field", "view": "sensor-shell"},
            )
            if item is not None:
                published.append(item)
        return tuple(published)


class GLPreviewCompositor:
    """Draw shared 2-D texture products without waiting for unfinished fences."""

    _VERT = """#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
    _FRAG = """#version 330 core
uniform sampler2D source_tex_2d;
uniform sampler2DArray source_tex_array;
uniform sampler3D source_tex_3d;
uniform int texture_kind;
uniform float source_slice;
uniform int flip_y;
uniform int alpha_mode;
uniform int tone_map;
uniform float exposure;
in vec2 uv;
out vec4 frag;
void main() {
    vec2 q = vec2(uv.x, flip_y != 0 ? 1.0 - uv.y : uv.y);
    vec4 v;
    if (texture_kind == 2)
        v = texture(source_tex_3d, vec3(q, source_slice));
    else if (texture_kind == 1)
        v = texture(source_tex_array, vec3(q, source_slice));
    else
        v = texture(source_tex_2d, q);
    vec3 rgb = max(v.rgb * exposure, vec3(0.0));
    if (tone_map != 0) rgb = rgb / (vec3(1.0) + rgb);
    if (alpha_mode == 2) v.a = 1.0;
    frag = vec4(rgb, clamp(v.a, 0.0, 1.0));
}
"""

    def __init__(self) -> None:
        self._program = 0
        self._vao = 0
        self._last_ready: dict[str, PreviewTextureProduct] = {}

    @staticmethod
    def _compile(gl: Any, shader_type: int, source: str) -> int:
        shader = int(gl.glCreateShader(shader_type))
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(shader).decode("utf-8", "replace"))
        return shader

    def init_gl(self) -> None:
        from OpenGL import GL as gl
        if self._program:
            return
        vertex = self._compile(gl, gl.GL_VERTEX_SHADER, self._VERT)
        fragment = self._compile(gl, gl.GL_FRAGMENT_SHADER, self._FRAG)
        program = int(gl.glCreateProgram())
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        gl.glDeleteShader(vertex)
        gl.glDeleteShader(fragment)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode("utf-8", "replace"))
        self._program = program
        self._vao = int(gl.glGenVertexArrays(1))

    @staticmethod
    def _ready(product: PreviewTextureProduct) -> bool:
        if product.sync_handle is None:
            return True
        from OpenGL import GL as gl
        result = gl.glClientWaitSync(product.sync_handle, 0, 0)
        return result in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED)

    def ready_product(
        self, product: PreviewTextureProduct | None,
    ) -> PreviewTextureProduct | None:
        if product is None:
            return None
        if self._ready(product):
            self._last_ready[product.product_id] = product
            return product
        return self._last_ready.get(product.product_id)

    def draw(
        self,
        product: PreviewTextureProduct | None,
        rect: tuple[int, int, int, int],
        framebuffer_height: int,
        *,
        exposure: float = 1.0,
        tone_map: bool = True,
    ) -> bool:
        product = self.ready_product(product)
        if product is None:
            return False
        from OpenGL import GL as gl
        self.init_gl()
        x, y, width, height = map(int, rect)
        if width <= 0 or height <= 0:
            return False
        old_viewport = tuple(int(v) for v in gl.glGetIntegerv(gl.GL_VIEWPORT))
        depth_enabled = bool(gl.glIsEnabled(gl.GL_DEPTH_TEST))
        blend_enabled = bool(gl.glIsEnabled(gl.GL_BLEND))
        try:
            gl.glViewport(x, int(framebuffer_height) - y - height, width, height)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)
            if product.alpha_mode == "premultiplied":
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
            else:
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glUseProgram(self._program)
            texture_kind = (
                2 if int(product.texture_target) == int(gl.GL_TEXTURE_3D)
                else 1 if int(product.texture_target) == int(gl.GL_TEXTURE_2D_ARRAY)
                else 0
            )
            texture_unit = texture_kind
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(int(product.texture_target), int(product.texture_id))
            gl.glUniform1i(gl.glGetUniformLocation(self._program, "source_tex_2d"), 0)
            gl.glUniform1i(gl.glGetUniformLocation(self._program, "source_tex_array"), 1)
            gl.glUniform1i(gl.glGetUniformLocation(self._program, "source_tex_3d"), 2)
            gl.glUniform1i(
                gl.glGetUniformLocation(self._program, "texture_kind"), texture_kind
            )
            raw_slice = float(product.metadata.get("slice", 0.5))
            if texture_kind == 1 and product.depth > 1:
                raw_slice = (float(product.metadata.get("layer", 0)) + 0.5) / product.depth
            gl.glUniform1f(
                gl.glGetUniformLocation(self._program, "source_slice"),
                max(0.0, min(1.0, raw_slice)),
            )
            gl.glUniform1i(
                gl.glGetUniformLocation(self._program, "flip_y"),
                1 if product.orientation == "top-left" else 0,
            )
            gl.glUniform1i(
                gl.glGetUniformLocation(self._program, "alpha_mode"),
                {"straight": 0, "premultiplied": 1, "opaque": 2}[product.alpha_mode],
            )
            gl.glUniform1i(gl.glGetUniformLocation(self._program, "tone_map"), int(tone_map))
            gl.glUniform1f(gl.glGetUniformLocation(self._program, "exposure"), float(exposure))
            gl.glBindVertexArray(self._vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            return True
        finally:
            gl.glBindVertexArray(0)
            gl.glBindTexture(int(product.texture_target), 0)
            gl.glUseProgram(0)
            if not blend_enabled:
                gl.glDisable(gl.GL_BLEND)
            if depth_enabled:
                gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glViewport(*old_viewport)

    def destroy(self) -> None:
        if not self._program and not self._vao:
            return
        from OpenGL import GL as gl
        if self._vao:
            gl.glDeleteVertexArrays(1, [self._vao])
        if self._program:
            gl.glDeleteProgram(self._program)
        self._vao = 0
        self._program = 0


@dataclass(slots=True)
class _CaptureSlot:
    pbo: int
    capacity: int = 0
    fence: Any = None
    product: PreviewTextureProduct | None = None


class AsyncTextureCapture:
    """PBO/fence capture whose encoding and filesystem work runs off-thread."""

    def __init__(self, output_dir: str, *, ring_size: int = 3) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.ring_size = max(2, int(ring_size))
        self._slots: list[_CaptureSlot] = []
        self._encoded: queue.Queue[tuple[PreviewTextureProduct, bytes]] = queue.Queue(
            maxsize=self.ring_size * 2
        )
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._write_loop, name="PreviewCaptureWriter", daemon=True
        )
        self._worker.start()
        self._captured: set[tuple[str, int]] = set()
        self.dropped = 0

    def init_gl(self) -> None:
        if self._slots:
            return
        from OpenGL import GL as gl
        raw = gl.glGenBuffers(self.ring_size)
        ids: Iterable[int] = (raw,) if self.ring_size == 1 else raw
        self._slots = [_CaptureSlot(int(item)) for item in ids]

    @staticmethod
    def _byte_size(product: PreviewTextureProduct) -> int:
        # Capture presentation-neutral float RGBA for scientific products.
        return int(product.width) * int(product.height) * int(product.depth) * 4 * 4

    def request(self, product: PreviewTextureProduct) -> bool:
        """Issue GPU→PBO DMA and return immediately; false means ring pressure."""
        key = (product.product_id, int(product.generation))
        if key in self._captured:
            return False
        from OpenGL import GL as gl
        self.init_gl()
        self.poll()
        slot = next((item for item in self._slots if item.product is None), None)
        if slot is None:
            self.dropped += 1
            return False
        size = self._byte_size(product)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, slot.pbo)
        if slot.capacity < size:
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, size, None, gl.GL_STREAM_READ)
            slot.capacity = size
        gl.glBindTexture(int(product.texture_target), int(product.texture_id))
        gl.glGetTexImage(
            int(product.texture_target), 0, gl.GL_RGBA, gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )
        gl.glBindTexture(int(product.texture_target), 0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        slot.fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        slot.product = replace(product, sync_handle=None)
        self._captured.add(key)
        return True

    def poll(self) -> int:
        """Move signalled PBOs to the writer queue; never wait for a fence."""
        if not self._slots:
            return 0
        from OpenGL import GL as gl
        completed = 0
        for slot in self._slots:
            if slot.product is None:
                continue
            state = gl.glClientWaitSync(slot.fence, 0, 0)
            if state not in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED):
                continue
            size = self._byte_size(slot.product)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, slot.pbo)
            pointer = gl.glMapBufferRange(
                gl.GL_PIXEL_PACK_BUFFER, 0, size, gl.GL_MAP_READ_BIT
            )
            payload = ctypes.string_at(pointer, size) if pointer else b""
            if pointer:
                gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            gl.glDeleteSync(slot.fence)
            if payload:
                try:
                    self._encoded.put_nowait((slot.product, payload))
                except queue.Full:
                    self.dropped += 1
            slot.fence = None
            slot.product = None
            completed += 1
        return completed

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "preview"

    def _write_loop(self) -> None:
        while not self._stop.is_set() or not self._encoded.empty():
            try:
                product, payload = self._encoded.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                import numpy as np
                os.makedirs(self.output_dir, exist_ok=True)
                stem = (
                    f"{self._safe_name(product.product_id)}_"
                    f"{int(product.generation):08d}"
                )
                image = np.frombuffer(payload, dtype=np.float32).reshape(
                    int(product.depth), int(product.height), int(product.width), 4
                )
                np.save(os.path.join(self.output_dir, stem + ".npy"), image)
                try:
                    from PIL import Image
                    layer = min(
                        int(product.depth) - 1,
                        max(0, int(product.metadata.get(
                            "layer", round(float(product.metadata.get("slice", 0.5))
                                           * max(0, int(product.depth) - 1)),
                        ))),
                    )
                    display_image = image[layer]
                    rgb = np.maximum(display_image[..., :3], 0.0)
                    rgb = rgb / (1.0 + rgb)
                    rgba8 = np.clip(
                        np.concatenate((rgb, np.clip(display_image[..., 3:4], 0.0, 1.0)), axis=2)
                        * 255.0, 0.0, 255.0,
                    ).astype(np.uint8)
                    Image.fromarray(rgba8[::-1], "RGBA").save(
                        os.path.join(self.output_dir, stem + ".png")
                    )
                except Exception:
                    pass
            finally:
                self._encoded.task_done()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=3.0)

    def destroy_gl(self) -> None:
        if not self._slots:
            return
        from OpenGL import GL as gl
        for slot in self._slots:
            if slot.fence is not None:
                gl.glDeleteSync(slot.fence)
        gl.glDeleteBuffers(len(self._slots), [slot.pbo for slot in self._slots])
        self._slots.clear()


__all__ = [
    "PreviewProductKind", "PreviewTextureProduct", "PreviewProductRegistry",
    "OpenGLShareGroup", "PreviewProductPublisher", "GLPreviewCompositor",
    "RayPipelinePreviewBridge", "AsyncTextureCapture",
    "LatestProductionResult", "LatestOnlyProducer",
]
