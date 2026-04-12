"""Lightweight line / dot / heatmap-bar plot widget for Pygame panels.

Renders simple 2-D plots onto a :class:`pygame.Surface` and supports:

* Multiple named data series with independent colours.
* Line mode, dot mode, or both.
* Automatic Y-axis range (or caller-supplied fixed range).
* Horizontal grid lines and optional axis labels.
* **Heatmap bar mode**: multiple 1-D (N, 3) float32 strips stacked
  vertically with labels and readability margins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

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
    clip_color: tuple[int, int, int] | None = None  # if set, segments outside clip range use this color


@dataclass
class PlotMarker:
    """A vertical marker line at a specific X value."""
    x_value: float
    label: str = ""
    color: tuple[int, int, int] = (160, 160, 160)
    dash: bool = True


@dataclass
class HeatmapBar:
    """One named heatmap strip — (N, 3) float32 in [0, 1].

    *data_fn* is an optional callable ``() -> (np.ndarray, np.ndarray)``
    returning ``(rgb, x)`` that will be called on each render to refresh.
    """
    key: str
    label: str
    rgb: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    data_fn: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None


# ---------------------------------------------------------------------------
# Plot widget
# ---------------------------------------------------------------------------

class PlotWidget:
    """Draws line series, dot series, and/or stacked heatmap bars.

    Coordinates are **surface-local** — the caller decides where on the
    parent surface this widget lives.

    **Heatmap-bar mode** (``self.bars``):

    Each :class:`HeatmapBar` is a 1-D colour strip.  They are stacked
    vertically with 1-px margins between them.  Left-side labels show
    bar names.  Bottom labels show the ``title`` text.  Vertical markers
    at the top with pitch labels.  If any bar has a ``data_fn``, it is
    called every render to refresh its data.

    **Line/dot mode** (``self.series``) works as before.
    """

    GRID_COLOR = (50, 50, 60)
    AXIS_COLOR = (100, 100, 110)
    BG_COLOR = (20, 20, 25, 220)
    MARGIN_LEFT = 36
    MARGIN_RIGHT = 6
    MARGIN_TOP = 14    # room for pitch labels at top
    MARGIN_BOTTOM = 14 # room for bottom text

    def __init__(self) -> None:
        self.series: list[PlotSeries] = []
        self.bars: list[HeatmapBar] = []
        self.markers: list[PlotMarker] = []
        self.title: str = ""
        self.y_min: float | None = None   # None → auto
        self.y_max: float | None = None
        self.grid_lines: int = 4
        self._font: pygame.font.Font | None = None
        # Clip boundary lines
        self.clip_lo: float | None = None
        self.clip_hi: float | None = None
        self.clip_boundary_color: tuple[int, int, int] = (200, 50, 50)
        # Legacy single-heatmap (kept for compat, but prefer bars)
        self.heatmap: np.ndarray | None = None
        self.heatmap_x: np.ndarray | None = None

    def _ensure_font(self) -> None:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("consolas", 11)

    def set_series(self, key: str, x: np.ndarray | Sequence[float],
                   y: np.ndarray | Sequence[float]) -> None:
        xa = np.asarray(x, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        for s in self.series:
            if s.key == key:
                s.data_x = xa
                s.data_y = ya
                return
        self.series.append(PlotSeries(key=key, label=key, data_x=xa, data_y=ya))

    def add_series(self, series: PlotSeries) -> None:
        self.series = [s for s in self.series if s.key != series.key]
        self.series.append(series)

    def add_bar(self, bar: HeatmapBar) -> None:
        self.bars = [b for b in self.bars if b.key != bar.key]
        self.bars.append(bar)

    def remove_bar(self, key: str) -> None:
        self.bars = [b for b in self.bars if b.key != key]

    def remove_series(self, key: str) -> None:
        self.series = [s for s in self.series if s.key != key]

    def clear(self) -> None:
        self.series.clear()
        self.bars.clear()

    # ----- heatmap-bar rendering -------------------------------------------

    @staticmethod
    def _make_strip(rgb: np.ndarray, target_w: int, target_h: int
                    ) -> pygame.Surface:
        """Scale an (N, 3) float32 array into a pygame Surface."""
        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        # pygame.surfarray needs (W, H, 3)
        arr = np.ascontiguousarray(rgb8[np.newaxis, :, :].transpose(1, 0, 2))
        raw = pygame.surfarray.make_surface(arr)
        return pygame.transform.scale(raw, (target_w, target_h))

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

        # --- refresh dynamic bars ---
        for bar in self.bars:
            if bar.data_fn is not None:
                try:
                    bar.rgb, bar.x = bar.data_fn()
                except Exception:
                    pass

        # Collect all X sources for range computation
        x_sources: list[tuple[float, float]] = []
        for s in self.series:
            if s.data_x.size:
                x_sources.append((float(s.data_x.min()),
                                  float(s.data_x.max())))
        for bar in self.bars:
            if bar.x.size:
                x_sources.append((float(bar.x.min()), float(bar.x.max())))
        if self.heatmap_x is not None and self.heatmap_x.size:
            x_sources.append((float(self.heatmap_x.min()),
                              float(self.heatmap_x.max())))
        if x_sources:
            xlo = min(lo for lo, _ in x_sources)
            xhi = max(hi for _, hi in x_sources)
        else:
            xlo, xhi = 0.0, 1.0
        if xhi - xlo < 1e-9:
            xhi = xlo + 1.0

        # --- multi-bar heatmap mode ---
        live_bars = [b for b in self.bars if b.rgb.shape[0] > 0]
        if live_bars:
            n_bars = len(live_bars)
            bar_gap = 1
            total_gaps = bar_gap * max(n_bars - 1, 0)
            bar_h = max(1, (inner_h - total_gaps) // n_bars)

            by = inner_y
            for bar in live_bars:
                strip = self._make_strip(bar.rgb, inner_w, bar_h)
                surf.blit(strip, (inner_x, by))
                # Left label
                lbl = font.render(bar.label, True, (160, 160, 160))
                ly = by + bar_h // 2 - lbl.get_height() // 2
                surf.blit(lbl, (x + 2, ly))
                by += bar_h + bar_gap

        # --- legacy single heatmap ---
        elif self.heatmap is not None and self.heatmap.shape[0] > 0:
            strip = self._make_strip(self.heatmap, inner_w, inner_h)
            surf.blit(strip, (inner_x, inner_y))

        # --- line/dot series (Y range needed) ---
        if self.series:
            ylo = self.y_min
            yhi = self.y_max
            if ylo is None or yhi is None:
                all_y = np.concatenate(
                    [s.data_y for s in self.series if s.data_y.size > 0]
                ) if any(s.data_y.size > 0 for s in self.series) \
                    else np.array([0.0, 1.0])
                if ylo is None:
                    ylo = float(all_y.min()) if all_y.size else 0.0
                if yhi is None:
                    yhi = float(all_y.max()) if all_y.size else 1.0
            if yhi - ylo < 1e-9:
                yhi = ylo + 1.0

            def to_px(vx: float, vy: float) -> tuple[int, int]:
                px = inner_x + int((vx - xlo) / (xhi - xlo) * inner_w)
                py = inner_y + inner_h - int((vy - ylo) / (yhi - ylo) * inner_h)
                return px, py

            # grid
            for gi in range(self.grid_lines + 1):
                frac = gi / max(self.grid_lines, 1)
                gy = inner_y + inner_h - int(frac * inner_h)
                pygame.draw.line(surf, self.GRID_COLOR,
                                 (inner_x, gy), (inner_x + inner_w, gy))
                val = ylo + frac * (yhi - ylo)
                lbl = font.render(f"{val:.2g}", True, self.AXIS_COLOR)
                surf.blit(lbl, (x + 2, gy - lbl.get_height() // 2))

            # clip boundary lines
            c_lo = self.clip_lo if self.clip_lo is not None else -float("inf")
            c_hi = self.clip_hi if self.clip_hi is not None else float("inf")
            if self.clip_lo is not None and ylo <= self.clip_lo <= yhi:
                _, cy = to_px(xlo, self.clip_lo)
                pygame.draw.line(surf, self.clip_boundary_color,
                                 (inner_x, cy), (inner_x + inner_w, cy), 1)
            if self.clip_hi is not None and ylo <= self.clip_hi <= yhi:
                _, cy = to_px(xlo, self.clip_hi)
                pygame.draw.line(surf, self.clip_boundary_color,
                                 (inner_x, cy), (inner_x + inner_w, cy), 1)

            # draw series
            for s in self.series:
                if s.data_x.size == 0:
                    continue
                pts = [to_px(float(s.data_x[i]), float(s.data_y[i]))
                       for i in range(len(s.data_x))]
                if s.line and len(pts) >= 2:
                    if s.clip_color is not None:
                        for j in range(len(pts) - 1):
                            y0_val = float(s.data_y[j])
                            y1_val = float(s.data_y[j + 1])
                            clipped = (y0_val < c_lo or y0_val > c_hi or
                                       y1_val < c_lo or y1_val > c_hi)
                            col = s.clip_color if clipped else s.color
                            pygame.draw.line(surf, col, pts[j], pts[j + 1], 1)
                    else:
                        pygame.draw.lines(surf, s.color, False, pts, 1)
                if s.dots:
                    for px, py in pts:
                        pygame.draw.circle(surf, s.color, (px, py),
                                           s.dot_radius)

        # --- vertical markers (top labels) ---
        for m in self.markers:
            if xlo <= m.x_value <= xhi:
                mx = inner_x + int((m.x_value - xlo) / (xhi - xlo) * inner_w)
                if m.dash:
                    seg = 4
                    for dy in range(inner_y, inner_y + inner_h, seg * 2):
                        y0 = dy
                        y1 = min(dy + seg, inner_y + inner_h)
                        pygame.draw.line(surf, m.color, (mx, y0), (mx, y1), 1)
                else:
                    pygame.draw.line(surf, m.color,
                                     (mx, inner_y), (mx, inner_y + inner_h), 1)
                if m.label and font:
                    mlbl = font.render(m.label, True, m.color)
                    lx_m = min(mx + 2,
                               inner_x + inner_w - mlbl.get_width())
                    surf.blit(mlbl, (lx_m, y + 1))  # top margin area

        # --- title (bottom) ---
        if self.title:
            ttl = font.render(self.title, True, (180, 180, 180))
            surf.blit(ttl, (inner_x, y + h - ttl.get_height() - 1))

        # --- legend (bottom right, for line series) ---
        if self.series and not self.bars:
            lx = inner_x
            for s in self.series:
                swatch_rect = pygame.Rect(lx, y + h - 13, 8, 8)
                pygame.draw.rect(surf, s.color, swatch_rect)
                lbl = font.render(s.label, True, (160, 160, 160))
                surf.blit(lbl, (lx + 11, y + h - 14))
                lx += 11 + lbl.get_width() + 8

        return h
