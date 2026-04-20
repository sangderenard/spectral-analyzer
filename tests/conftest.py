"""conftest.py — ensure bass_viewer is importable in test environments.

pygame / PyOpenGL may not be installed in headless CI.  We stub them
just enough that ``import bass_viewer`` succeeds without a display.
"""

from __future__ import annotations

import sys
import types
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_module_stub(name: str) -> None:
    """Insert a minimal stub for *name* (and sub-modules) if missing."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    sys.modules[name] = mod


# ---- pygame stubs ----
_pg_submodules = [
    "pygame", "pygame.locals", "pygame.mixer", "pygame.font",
    "pygame.draw", "pygame.transform", "pygame.surfarray",
    "pygame.sndarray", "pygame.image", "pygame.display",
    "pygame.event", "pygame.time", "pygame.key", "pygame.mouse",
    "pygame.cursors",
]
for _name in _pg_submodules:
    _ensure_module_stub(_name)

# pygame.locals must export the constants bass_viewer imports
_locals = sys.modules["pygame.locals"]
_needed_consts = [
    "DOUBLEBUF", "KEYDOWN", "KEYUP", "MOUSEBUTTONDOWN", "MOUSEBUTTONUP",
    "MOUSEMOTION", "MOUSEWHEEL", "OPENGL", "QUIT", "RESIZABLE", "VIDEORESIZE",
    "K_SPACE", "K_ESCAPE", "K_LEFT", "K_RIGHT", "K_HOME", "K_END", "K_TAB",
    "K_r", "K_f", "K_t", "K_x", "K_y", "K_s", "K_a", "K_d",
    "K_LSHIFT", "K_RSHIFT",
    # analytic_driver additional keys
    "K_DELETE", "K_n", "K_o", "K_z", "K_LCTRL", "K_RCTRL",
]
for _c in _needed_consts:
    setattr(_locals, _c, 0)

# KMOD constants live on the pygame module itself (not pygame.locals) in pygame 2.x
_kmod_consts = [
    "KMOD_CTRL", "KMOD_SHIFT", "KMOD_ALT", "KMOD_NONE",
    "KMOD_LSHIFT", "KMOD_RSHIFT", "KMOD_LCTRL", "KMOD_RCTRL",
]

# pygame itself needs Surface, Rect, etc.
_pg = sys.modules["pygame"]
_pg.Surface = type("Surface", (), {"__init__": lambda *a, **kw: None})
_pg.Rect = type("Rect", (), {"__init__": lambda *a, **kw: None})
_pg.DOUBLEBUF = 0
_pg.OPENGL = 0
_pg.RESIZABLE = 0
_pg.mixer = sys.modules["pygame.mixer"]
_pg.mixer.Channel = type("Channel", (), {})
_pg.mixer.Sound = type("Sound", (), {})
_pg.font = sys.modules["pygame.font"]
_pg.font.Font = type("Font", (), {})
_pg.font.SysFont = lambda *a, **kw: None
# KMOD constants — must live on the pygame module itself
for _c in _kmod_consts:
    setattr(_pg, _c, 0)

# ---- OpenGL stubs ----
for _gl_name in [
    "OpenGL", "OpenGL.GL", "OpenGL.GL.shaders",
    "OpenGL.GLU", "OpenGL.GLUT",
]:
    _ensure_module_stub(_gl_name)

# Make `from OpenGL.GL import *` produce nothing harmful
_gl_mod = sys.modules["OpenGL.GL"]
_gl_mod.__all__ = []
# Provide the shaders alias
_gl_mod.shaders = sys.modules["OpenGL.GL.shaders"]
