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

# ---- scipy availability ----
try:
    import scipy.signal as _scipy_signal  # type: ignore
except Exception:
    for _scipy_name in ["scipy", "scipy.signal"]:
        _ensure_module_stub(_scipy_name)
    _scipy_signal = sys.modules["scipy.signal"]

if not hasattr(_scipy_signal, "resample_poly"):
    def _resample_poly_stub(x, up, down, *_args, **_kwargs):
        import numpy as _np
        arr = _np.asarray(x)
        up_i = max(1, int(up))
        down_i = max(1, int(down))
        if up_i == down_i:
            return arr
        out_len = max(1, int(round(len(arr) * up_i / down_i)))
        if len(arr) == 0:
            return arr
        idx = _np.linspace(0, len(arr) - 1, out_len)
        idx = _np.clip(_np.round(idx).astype(int), 0, len(arr) - 1)
        return arr[idx]
    _scipy_signal.resample_poly = _resample_poly_stub

# ---- viewer dependency availability used by analytic_driver tests ----
try:
    import plot_widget as _plot_widget  # type: ignore
except Exception:
    _ensure_module_stub("plot_widget")
    _plot_widget = sys.modules["plot_widget"]

    class _PlotSeries:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _PlotMarker:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _PlotWidget:
        def __init__(self, *args, **kwargs):
            self.series = []
            self.markers = []
            self.y_min = -1.0
            self.y_max = 1.0
            self.grid_lines = 0

        def add_series(self, series):
            self.series.append(series)

        def render(self, *args, **kwargs):
            return None

    _plot_widget.PlotWidget = _PlotWidget
    _plot_widget.PlotSeries = _PlotSeries
    _plot_widget.PlotMarker = _PlotMarker

try:
    import bass_viewer as _bass_viewer  # type: ignore
except Exception:
    _ensure_module_stub("bass_viewer")
    _bass_viewer = sys.modules["bass_viewer"]
    for _name in [
        "GlyphAtlas", "Panel", "PanelDock", "ScrollableSubpanelList",
        "ModularSubpanelSpec", "SubpanelAddOption", "FilterBankDecomposition",
    ]:
        if not hasattr(_bass_viewer, _name):
            setattr(_bass_viewer, _name, type(_name, (), {}))
