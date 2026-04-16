"""engine_reporter.py — unified progress + timing reporter for analysis engines.

Usage inside each _execute_*_analysis_run method::

    reporter = EngineRunReporter(progress_cb, run_label=run.label)
    reporter.phase("load")
    # ... load audio ...
    reporter.phase("transform", n_total=n_batches)
    for i, batch in enumerate(batches):
        # ... process ...
        reporter.batch(n_items=len(batch))
        reporter.progress(i / n_batches, f"transform {i}/{n_batches}")
    reporter.phase("save")
    # ... save ...
    stats = reporter.finalize()   # closes last phase, returns stats dict
    run.metadata["stats"] = stats
"""

from __future__ import annotations

import time
from typing import Callable


ProgressCallback = Callable[[float, str], None]


class EngineRunReporter:
    """Wraps a progress callback and accumulates timing + batch stats per phase.

    Thread-safety: all methods are called from the worker thread; no locking
    needed as long as the caller itself is single-threaded.
    """

    def __init__(self, progress_cb: ProgressCallback, run_label: str = "") -> None:
        self._cb = progress_cb
        self._label = run_label
        self._t_start = time.perf_counter()
        self._phases: list[dict] = []
        self._cur: dict | None = None
        self._n_batches_total = 0
        self._n_items_total = 0

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def phase(self, name: str, *, n_total: int = 0) -> None:
        """Start a new named phase (closes the previous one if open).

        *n_total* is an optional expected item count for the phase
        (purely informational; stored in the stats output).
        """
        self._close_phase()
        self._cur = {
            "name": name,
            "t_start": time.perf_counter(),
            "n_batches": 0,
            "n_items": 0,
            "n_total": n_total,
            "last_label": "",
        }

    def _close_phase(self) -> None:
        if self._cur is None:
            return
        elapsed = time.perf_counter() - self._cur.pop("t_start")
        self._cur["elapsed_sec"] = round(elapsed, 4)
        self._phases.append(self._cur)
        self._cur = None

    # ------------------------------------------------------------------
    # Per-batch accounting
    # ------------------------------------------------------------------

    def batch(self, n_items: int = 1) -> None:
        """Record completion of one batch containing *n_items* units."""
        self._n_batches_total += 1
        self._n_items_total += n_items
        if self._cur is not None:
            self._cur["n_batches"] += 1
            self._cur["n_items"] += n_items

    # ------------------------------------------------------------------
    # Progress forwarding
    # ------------------------------------------------------------------

    def progress(self, frac: float, label: str) -> None:
        """Forward (*frac*, *label*) to the underlying callback and record
        the most recent label for the current phase."""
        if self._cur is not None:
            self._cur["last_label"] = label
        self._cb(frac, label)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        """Close the current phase and return a serialisable stats dict.

        The dict is safe to store in ``run.metadata["stats"]``.
        """
        self._close_phase()
        total_elapsed = time.perf_counter() - self._t_start
        phases_out = []
        for p in self._phases:
            entry: dict = {
                "name": p["name"],
                "elapsed_sec": p["elapsed_sec"],
                "n_batches": p["n_batches"],
                "n_items": p["n_items"],
            }
            if p.get("n_total"):
                entry["n_total"] = p["n_total"]
            if p.get("last_label"):
                entry["last_label"] = p["last_label"]
            phases_out.append(entry)
        stats = {
            "engine_label": self._label,
            "total_elapsed_sec": round(total_elapsed, 4),
            "n_batches": self._n_batches_total,
            "n_items": self._n_items_total,
            "phases": phases_out,
        }
        _print_run_stats(stats)
        return stats


# ---------------------------------------------------------------------------
# Console summary helper
# ---------------------------------------------------------------------------

def _print_run_stats(stats: dict) -> None:
    """Print a compact timing + batch summary to stdout."""
    label = stats.get("engine_label") or "run"
    total = stats.get("total_elapsed_sec", 0.0)
    n_items = stats.get("n_items", 0)
    phases = stats.get("phases", [])
    lines = [f"[stats] {label}  total={total:.2f}s  items={n_items:,}"]
    for p in phases:
        ph_name = p["name"]
        ph_sec = p.get("elapsed_sec", 0.0)
        ph_items = p.get("n_items", 0)
        ph_batches = p.get("n_batches", 0)
        detail = f"{ph_sec:.2f}s"
        if ph_batches:
            detail += f"  {ph_batches}b/{ph_items}i"
        lines.append(f"  {ph_name:<16s} {detail}")
    print("\n".join(lines), flush=True)
