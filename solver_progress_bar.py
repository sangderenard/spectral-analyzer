"""solver_progress_bar.py — Modular progress/statistics bar for GraphSolver.

Surface-based widget compatible with ParametricCurveEditor, PlotWidget, and
any other sub-unit panel in bass_viewer / analytic_driver.  No analytic_driver
imports — bind to any GraphSolver + optional EdgeFifoBank.

Usage
─────
    bar = SolverProgressBar(w=800, h=72)
    bar.bind(solver, fifo_bank=bank)          # attach once after compile

    # pass the callback into run_schedule:
    outputs = solver.run_schedule({}, n_frames=N,
                                  on_progress=bar.make_progress_callback())

    # in the render loop:
    bar.render(surf, x=0, y=surf.get_height() - bar.h, font=font)

Thread safety
─────────────
    Progress updates arrive from a background render thread.
    All mutable state is protected by an internal Lock.
    render() is safe to call from the main (pygame) thread at any time.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (same dark-purple family used in analytic_driver panels)
# ─────────────────────────────────────────────────────────────────────────────
_C_BG       = (18,  18,  24)
_C_TRACK    = (38,  32,  54)
_C_BAR_IDLE = (55,  45,  80)
_C_BAR_RUN  = (110, 70, 200)
_C_BAR_DONE = (70, 180,  90)
_C_BAR_ERR  = (200, 60,  60)
_C_BORDER   = (70,  60,  95)
_C_TXT      = (200, 190, 220)
_C_DIM      = (110, 100, 130)
_C_ACCENT   = (160, 120, 255)
_C_FIFO     = (80,  180, 160)


# ─────────────────────────────────────────────────────────────────────────────
# SolverProgressBar
# ─────────────────────────────────────────────────────────────────────────────

class SolverProgressBar:
    """Progress + statistics overlay for a GraphSolver render.

    Parameters
    ----------
    w, h:
        Pixel dimensions of the rendered surface region.
    fifo_slot_labels:
        Optional mapping from fifo slot key → short display label.
        If omitted, raw slot keys are displayed.
    """

    def __init__(
        self,
        w: int = 800,
        h: int = 72,
        fifo_slot_labels: Optional[dict] = None,
    ) -> None:
        self.w = w
        self.h = h
        self._slot_labels: dict = dict(fifo_slot_labels or {})

        self._lock = threading.Lock()

        # Graph-structural snapshot (set once by bind)
        self._graph_stats: dict = {}

        # Live state (updated from render thread via progress callback)
        self._progress: tuple[int, int] = (0, 1)   # (completed, total) plan items
        self._last_label: str = ""
        self._status: str = "ready"
        self._render_time_s: float = 0.0
        self._render_start: float = 0.0
        self._fifo_snapshot: dict[str, int] = {}    # slot → count at snapshot

        # References held weakly-enough to never block GC
        self._solver: Any = None
        self._fifo_bank: Any = None

    # ── Binding ──────────────────────────────────────────────────────────────

    def bind(self, solver: Any, fifo_bank: Any = None) -> None:
        """Attach to a compiled GraphSolver (and optional EdgeFifoBank).

        Call once after compile_nodes / GraphSolver construction.
        """
        with self._lock:
            self._solver    = solver
            self._fifo_bank = fifo_bank
            if hasattr(solver, "graph_stats"):
                self._graph_stats = solver.graph_stats()
            self._progress = (0, max(1, self._graph_stats.get("plan_items", 1)))

    # ── Status control ───────────────────────────────────────────────────────

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def start_render(self) -> None:
        """Call just before solver.run_schedule() to start the timing clock."""
        with self._lock:
            self._render_start = time.perf_counter()
            self._progress = (0, max(1, self._graph_stats.get("plan_items", 1)))
            self._status   = "rendering\u2026"

    def finish_render(self, success: bool = True, label: str = "") -> None:
        """Call after solver.run_schedule() completes."""
        with self._lock:
            elapsed = time.perf_counter() - self._render_start
            self._render_time_s = elapsed
            total = max(1, self._graph_stats.get("plan_items", 1))
            self._progress = (total, total)
            self._status   = label if label else ("done" if success else "error")
            self._snapshot_fifo()

    # ── Progress callback factory ─────────────────────────────────────────────

    def make_progress_callback(self) -> Callable[[int, int, str], None]:
        """Return a callable suitable for ``solver.run_schedule(on_progress=...)``.

        The callback is thread-safe and can be called from the render thread.
        """
        bar = self  # closure

        def _cb(completed: int, total: int, label: str) -> None:
            with bar._lock:
                bar._progress   = (completed, max(1, total))
                bar._last_label = label

        return _cb

    # ── Render ───────────────────────────────────────────────────────────────

    def render(
        self,
        surf: Any,
        x: int = 0,
        y: int = 0,
        font: Any = None,
    ) -> None:
        """Draw the progress bar onto *surf* at pixel offset (x, y).

        Safe to call from the main thread at any time; takes a snapshot of
        all mutable state under the lock before drawing.
        """
        try:
            import pygame
        except ImportError:
            return

        with self._lock:
            progress    = self._progress
            status      = self._status
            last_label  = self._last_label
            graph_stats = dict(self._graph_stats)
            fifo_snap   = dict(self._fifo_snapshot)
            render_time = self._render_time_s

        w, h = self.w, self.h
        if font is None:
            try:
                font = pygame.font.SysFont("consolas", 11)
            except Exception:
                return

        fh = font.get_height()
        pad = 6

        # ── Background ───────────────────────────────────────────────────────
        bg_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, _C_BG, bg_rect)
        pygame.draw.rect(surf, _C_BORDER, bg_rect, 1)

        # ── Row 0: status + graph stats ──────────────────────────────────────
        row0_y = y + pad

        status_color = (
            _C_BAR_ERR  if "error" in status else
            _C_BAR_DONE if status in ("done", "ready") else
            _C_BAR_RUN
        )
        surf.blit(
            font.render(status.upper(), True, status_color),
            (x + pad, row0_y),
        )

        gs_parts = []
        if graph_stats.get("nodes"):
            gs_parts.append(f"{graph_stats['nodes']} nodes")
        if graph_stats.get("edges"):
            gs_parts.append(f"{graph_stats['edges']} edges")
        n_plan = graph_stats.get("plan_items", 0)
        tier   = graph_stats.get("condensed_tier")
        if n_plan:
            tier_tag = f"T{tier}" if tier else ""
            gs_parts.append(f"{n_plan} steps {tier_tag}".strip())
        if graph_stats.get("contract_edges"):
            gs_parts.append(f"{graph_stats['contract_edges']} contracts")

        gs_txt = "  ·  ".join(gs_parts)
        gs_surf = font.render(gs_txt, True, _C_DIM)
        surf.blit(gs_surf, (x + w - gs_surf.get_width() - pad, row0_y))

        # ── Row 1: progress bar ───────────────────────────────────────────────
        row1_y = row0_y + fh + 4
        bar_h  = max(8, fh - 2)
        bar_x  = x + pad
        bar_w  = w - 2 * pad

        pygame.draw.rect(surf, _C_TRACK, pygame.Rect(bar_x, row1_y, bar_w, bar_h),
                         border_radius=3)

        completed, total = progress
        frac = completed / max(total, 1)
        filled = max(0, int(bar_w * frac))
        if filled > 0:
            bar_color = (
                _C_BAR_ERR  if "error" in status else
                _C_BAR_DONE if frac >= 1.0 else
                _C_BAR_RUN
            )
            pygame.draw.rect(surf, bar_color,
                             pygame.Rect(bar_x, row1_y, filled, bar_h),
                             border_radius=3)

        pct_txt = f"{int(frac * 100):3d}%"
        pct_surf = font.render(pct_txt, True, _C_TXT)
        surf.blit(pct_surf, (x + w - pct_surf.get_width() - pad, row1_y))

        # Last SCC label (truncated)
        if last_label:
            max_lbl_w = bar_w - pct_surf.get_width() - 8
            lbl_surf = font.render(last_label, True, _C_DIM)
            if lbl_surf.get_width() > max_lbl_w:
                # Truncate label to fit
                chars = max(4, int(max_lbl_w / max(1, lbl_surf.get_width()) * len(last_label)))
                lbl_surf = font.render(last_label[:chars] + "\u2026", True, _C_DIM)
            surf.blit(lbl_surf, (bar_x + 4, row1_y))

        # ── Row 2: fifo stats + render time ──────────────────────────────────
        row2_y = row1_y + bar_h + 4

        if fifo_snap:
            parts = []
            for slot_key, count in fifo_snap.items():
                lbl = self._slot_labels.get(slot_key, slot_key.split("_")[-2]
                                             if "_" in slot_key else slot_key)
                parts.append(f"{lbl}: {count}")
            fifo_txt = "  ".join(parts)
            surf.blit(font.render(fifo_txt, True, _C_FIFO), (x + pad, row2_y))

        if render_time > 0.0:
            rt_txt = f"render {render_time:.2f}s"
            rt_surf = font.render(rt_txt, True, _C_DIM)
            surf.blit(rt_surf, (x + w - rt_surf.get_width() - pad, row2_y))

    # ── Internal ─────────────────────────────────────────────────────────────

    def _snapshot_fifo(self) -> None:
        """Capture fifo slot counts.  Must be called under self._lock."""
        bank = self._fifo_bank
        if bank is None:
            return
        slot_keys_fn = getattr(bank, "slot_keys", None)
        count_fn     = getattr(bank, "count", None)
        if not callable(slot_keys_fn) or not callable(count_fn):
            return
        snap = {}
        for key in slot_keys_fn():
            try:
                snap[key] = int(count_fn(key))
            except Exception:
                pass
        self._fifo_snapshot = snap
