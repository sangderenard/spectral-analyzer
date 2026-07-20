"""Explicit parent-or-owned OpenGL context lifecycle for manifest widgets."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .control_layout import OpenGLContextRequest


class OpenGLContextSource(str, Enum):
    PARENT = "parent"
    CURRENT = "current"
    OWNED = "owned"
    UNAVAILABLE = "unavailable"


@dataclass
class OpenGLContextLease:
    context: Any = None
    source: OpenGLContextSource = OpenGLContextSource.UNAVAILABLE
    error: str = ""
    _release: Callable[[], None] | None = None
    _closed: bool = False

    @property
    def available(self) -> bool:
        return self.source is not OpenGLContextSource.UNAVAILABLE

    @property
    def owned(self) -> bool:
        return self.source is OpenGLContextSource.OWNED

    def activate(self) -> None:
        """Make this lease current when its backend exposes that operation."""

        make_current = getattr(self.context, "make_current", None)
        if callable(make_current):
            make_current()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.owned and self._release is not None:
            self._release()

    def __enter__(self) -> "OpenGLContextLease":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class _SDLOpenGLContext:
    """Independent hidden SDL window plus an explicitly current GL context."""

    SDL_WINDOWPOS_UNDEFINED = 0x1FFF0000
    SDL_WINDOW_OPENGL = 0x00000002
    SDL_WINDOW_HIDDEN = 0x00000008
    SDL_GL_CONTEXT_MAJOR_VERSION = 17
    SDL_GL_CONTEXT_MINOR_VERSION = 18

    def __init__(self, request: OpenGLContextRequest) -> None:
        import pygame

        pygame.display.init()
        candidates = [
            os.path.join(os.path.dirname(pygame.__file__), "SDL2.dll"),
            ctypes.util.find_library("SDL2-2.0"),
            ctypes.util.find_library("SDL2"),
        ]
        library_path = next(
            (candidate for candidate in candidates if candidate and (
                os.path.isfile(candidate) or candidate == os.path.basename(candidate)
            )),
            None,
        )
        if not library_path:
            raise RuntimeError("SDL2 library was not found for an owned GL context")
        self._sdl = ctypes.CDLL(library_path)
        self._configure_symbols()
        major, minor = map(int, request.minimum_version)
        self._sdl.SDL_GL_SetAttribute(self.SDL_GL_CONTEXT_MAJOR_VERSION, major)
        self._sdl.SDL_GL_SetAttribute(self.SDL_GL_CONTEXT_MINOR_VERSION, minor)
        flags = self.SDL_WINDOW_OPENGL | self.SDL_WINDOW_HIDDEN
        self._window = self._sdl.SDL_CreateWindow(
            b"manifest-opengl-widget",
            self.SDL_WINDOWPOS_UNDEFINED,
            self.SDL_WINDOWPOS_UNDEFINED,
            16,
            16,
            flags,
        )
        if not self._window:
            raise RuntimeError(self._error("SDL_CreateWindow"))
        self._context = self._sdl.SDL_GL_CreateContext(self._window)
        if not self._context:
            self._sdl.SDL_DestroyWindow(self._window)
            self._window = None
            raise RuntimeError(self._error("SDL_GL_CreateContext"))
        self.make_current()

    def _configure_symbols(self) -> None:
        self._sdl.SDL_GetError.restype = ctypes.c_char_p
        self._sdl.SDL_GL_SetAttribute.argtypes = [ctypes.c_int, ctypes.c_int]
        self._sdl.SDL_GL_SetAttribute.restype = ctypes.c_int
        self._sdl.SDL_CreateWindow.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
        ]
        self._sdl.SDL_CreateWindow.restype = ctypes.c_void_p
        self._sdl.SDL_GL_CreateContext.argtypes = [ctypes.c_void_p]
        self._sdl.SDL_GL_CreateContext.restype = ctypes.c_void_p
        self._sdl.SDL_GL_MakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._sdl.SDL_GL_MakeCurrent.restype = ctypes.c_int
        self._sdl.SDL_GL_DeleteContext.argtypes = [ctypes.c_void_p]
        self._sdl.SDL_DestroyWindow.argtypes = [ctypes.c_void_p]

    def _error(self, operation: str) -> str:
        raw = self._sdl.SDL_GetError()
        detail = raw.decode("utf-8", "replace") if raw else "unknown SDL error"
        return f"{operation} failed: {detail}"

    def make_current(self) -> None:
        if not self._window or not self._context:
            raise RuntimeError("owned OpenGL context is closed")
        if self._sdl.SDL_GL_MakeCurrent(self._window, self._context) != 0:
            raise RuntimeError(self._error("SDL_GL_MakeCurrent"))

    def close(self) -> None:
        if self._context:
            self._sdl.SDL_GL_DeleteContext(self._context)
            self._context = None
        if self._window:
            self._sdl.SDL_DestroyWindow(self._window)
            self._window = None


class OpenGLContextHost:
    """Borrow a parent/current GL context or create and own a fallback.

    ``owned_factory`` returns either a context or ``(context, release)``. This
    keeps context creation backend-neutral: an SDL, Qt, EGL, or WGL host can
    provide its native factory without the widget knowing which window system
    owns it. The default uses a separate hidden SDL window, so it can coexist
    with a parent software window without replacing that display.
    """

    def __init__(
        self,
        parent_context: Any = None,
        *,
        owned_factory: Callable[[OpenGLContextRequest], Any] | None = None,
        current_probe: Callable[[], Any] | None = None,
    ) -> None:
        self.parent_context = parent_context
        self.owned_factory = owned_factory
        self.current_probe = current_probe or self._probe_current_context
        self._leases: list[OpenGLContextLease] = []

    @staticmethod
    def _probe_current_context() -> Any:
        try:
            from OpenGL.GL import GL_VERSION, glGetString

            return glGetString(GL_VERSION)
        except Exception:
            return None

    @staticmethod
    def _default_owned_factory(request: OpenGLContextRequest) -> Any:
        context = _SDLOpenGLContext(request)
        return context, context.close
    def acquire(self, request: OpenGLContextRequest) -> OpenGLContextLease:
        if request.prefer_parent and self.parent_context is not None:
            lease = OpenGLContextLease(
                self.parent_context, OpenGLContextSource.PARENT
            )
            self._leases.append(lease)
            return lease
        current = self.current_probe()
        if current is not None:
            lease = OpenGLContextLease(current, OpenGLContextSource.CURRENT)
            self._leases.append(lease)
            return lease
        if not request.create_if_missing:
            lease = OpenGLContextLease(
                source=OpenGLContextSource.UNAVAILABLE,
                error="no parent or current OpenGL context",
            )
            self._leases.append(lease)
            return lease
        factory = self.owned_factory or self._default_owned_factory
        try:
            created = factory(request)
            if isinstance(created, tuple) and len(created) == 2:
                context, release = created
            else:
                context, release = created, None
            lease = OpenGLContextLease(
                context, OpenGLContextSource.OWNED, _release=release
            )
        except Exception as exc:
            lease = OpenGLContextLease(
                source=OpenGLContextSource.UNAVAILABLE,
                error=str(exc),
            )
        self._leases.append(lease)
        return lease

    def close(self) -> None:
        for lease in reversed(self._leases):
            lease.close()
        self._leases.clear()

    def __enter__(self) -> "OpenGLContextHost":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = [
    "OpenGLContextSource",
    "OpenGLContextLease",
    "OpenGLContextHost",
]