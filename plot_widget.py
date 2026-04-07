"""Lightweight line / dot plot widget for embedding inside Pygame panels.

Renders simple 2-D plots onto a :class:`pygame.Surface` and supports:

* Multiple named data series with independent colours.
* Line mode, dot mode, or both.
* Automatic Y-axis range (or caller-supplied fixed range).
* Horizontal grid lines and optional axis labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pygame


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PlotSeries:
    """One named data series."""
    key: str
    label: str
    color: tuple[int, int, int] = (200, 200, 200)
    line: bool = True
    dots: bool = False
    dot_radius: int = 2
    data_x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    data_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


# ---------------------------------------------------------------------------
# Plot widget
# ---------------------------------------------------------------------------

class PlotWidget:
    """Draws one or more :class:`PlotSeries` onto a caller-provided surface.

    Coordinates are **surface-local** — the caller decides where on the
    parent surface this widget lives.
    """

    GRID_COLOR = (50, 50, 60)
    AXIS_COLOR = (100, 100, 110)
    BG_COLOR = (20, 20, 25, 220)
    MARGIN_LEFT = 36
    MARGIN_RIGHT = 6
    MARGIN_TOP = 4
    MARGIN_BOTTOM = 16

    def __init__(self) -> None:
        self.series: list[PlotSeries] = []
        self.title: str = ""
        self.y_min: float | None = None   # None → auto
        self.y_max: float | None = None
        self.grid_lines: int = 4
        self._font: pygame.font.Font | None = None

    def _ensure_font(self) -> None:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 11)

    def set_series(self, key: str, x: np.ndarray | Sequence[float],
                   y: np.ndarray | Sequence[float]) -> None:
        """Update (or create) data for the series identified by *key*."""
        xa = np.asarray(x, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        for s in self.series:
            if s.key == key:
                s.data_x = xa
                s.data_y = ya
                return
        self.series.append(PlotSeries(key=key, label=key, data_x=xa, data_y=ya))

    def add_series(self, series: PlotSeries) -> None:
        """Add a fully configured series."""
        self.series = [s for s in self.series if s.key != series.key]
        self.series.append(series)

    def remove_series(self, key: str) -> None:
        self.series = [s for s in self.series if s.key != key]

    def clear(self) -> None:
        self.series.clear()

    # ----- rendering -------------------------------------------------------

    def render(self, surf: pygame.Surface, x: int, y: int,
               w: int, h: int, font: pygame.font.Font | None = None) -> int:
        """Draw the plot box at *(x, y)* with size *w × h* on *surf*.

        Returns the height consumed (always *h*).
        """
        if font is None:
            self._ensure_font()
            font = self._font

        plot_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, self.BG_COLOR, plot_rect)
        pygame.draw.rect(surf, self.AXIS_COLOR, plot_rect, 1)

        inner_x = x + self.MARGIN_LEFT
        inner_y = y + self.MARGIN_TOP
        inner_w = w - self.MARGIN_LEFT - self.MARGIN_RIGHT
        inner_h = h - self.MARGIN_TOP - self.MARGIN_BOTTOM

        if inner_w < 2 or inner_h < 2:
            return h

        # --- compute Y range ---
        ylo = self.y_min
        yhi = self.y_max
        if ylo is None or yhi is None:
            all_y = np.concatenate([s.data_y for s in self.series
                                    if s.data_y.size > 0])  \
                    if any(s.data_y.size > 0 for s in self.series) \
                    else np.array([0.0, 1.0])
            if ylo is None:
                ylo = float(all_y.min()) if all_y.size else 0.0
            if yhi is None:
                yhi = float(all_y.max()) if all_y.size else 1.0
        if yhi - ylo < 1e-9:
            yhi = ylo + 1.0

        # --- compute X range ---
        all_x_min = min((s.data_x.min() for s in self.series if s.data_x.size),
                        default=0.0)
        all_x_max = max((s.data_x.max() for s in self.series if s.data_x.size),
                        default=1.0)
        xlo, xhi = float(all_x_min), float(all_x_max)
        if xhi - xlo < 1e-9:
            xhi = xlo + 1.0

        def to_px(vx: float, vy: float) -> tuple[int, int]:
            px = inner_x + int((vx - xlo) / (xhi - xlo) * inner_w)
            py = inner_y + inner_h - int((vy - ylo) / (yhi - ylo) * inner_h)
            return px, py

        # --- grid ---
        for gi in range(self.grid_lines + 1):
            frac = gi / max(self.grid_lines, 1)
            gy = inner_y + inner_h - int(frac * inner_h)
            pygame.draw.line(surf, self.GRID_COLOR,
                             (inner_x, gy), (inner_x + inner_w, gy))
            val = ylo + frac * (yhi - ylo)
            lbl = font.render(f"{val:.2g}", True, self.AXIS_COLOR)
            surf.blit(lbl, (x + 2, gy - lbl.get_height() // 2))

        # --- series ---
        for s in self.series:
            if s.data_x.size == 0:
                continue
            pts = [to_px(float(s.data_x[i]), float(s.data_y[i]))
                   for i in range(len(s.data_x))]
            if s.line and len(pts) >= 2:
                pygame.draw.lines(surf, s.color, False, pts, 1)
            if s.dots:
                for px, py in pts:
                    pygame.draw.circle(surf, s.color, (px, py), s.dot_radius)

        # --- title ---
        if self.title:
            ttl = font.render(self.title, True, (180, 180, 180))
            surf.blit(ttl, (x + w // 2 - ttl.get_width() // 2, y + 2))

        # --- legend (compact, bottom) ---
        lx = inner_x
        for s in self.series:
            swatch_rect = pygame.Rect(lx, y + h - 13, 8, 8)
            pygame.draw.rect(surf, s.color, swatch_rect)
            lbl = font.render(s.label, True, (160, 160, 160))
            surf.blit(lbl, (lx + 11, y + h - 14))
            lx += 11 + lbl.get_width() + 8

        return h
