"""simulator_workspace.py
=========================
SimulatorWorkspace — state machine for the simulator station.

Follows the same pattern as FabricatorWorkspace: pure logic, no GL.

States
------
IDLE        Station is powered on but no plugin is running.
RUNNING     A plugin is active and stepping forward each frame.
PAUSED      Plugin step is suspended; parameters may be adjusted.
EVOLVING    CoevolutionScheduler is driving parameter exploration.

The workspace:
  * holds the plugin registry (loaded from simulator_palette.yaml)
  * owns the CoevolutionScheduler
  * exposes the current plugin's active parameters and state
  * is polled by SimulatorStation (renderer) for display data
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

from sm_plugins._base import SimulatorPlugin, load_plugin
from coevolution_scheduler import CoevolutionScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Mode enum
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorMode(Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    PAUSED   = "paused"
    EVOLVING = "evolving"


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorWorkspace:
    """Simulator state machine.  Pure logic, no GL.

    Parameters
    ----------
    palette_cfg:
        Parsed content of simulator_palette.yaml — ``{"plugins": [...], ...}``.
    coevo_cfg:
        Parsed content of coevolution.yaml.
    n_items:
        Number of simulation items (e.g. strings, particles) per plugin.
    use_torch:
        Backend preference passed through to plugin ``step()``.
    """

    def __init__(
        self,
        palette_cfg:  dict,
        coevo_cfg:    dict,
        n_items:      int  = 4,
        use_torch:    bool = False,
    ) -> None:
        self._palette_cfg = palette_cfg
        self._coevo_cfg   = coevo_cfg
        self.n_items      = n_items
        self.use_torch    = use_torch

        # ── Mode ──────────────────────────────────────────────────────────────
        self.mode = SimulatorMode.IDLE

        # ── Plugin registry ───────────────────────────────────────────────────
        # Loaded lazily when the user selects a plugin to avoid import cost.
        self._plugin_defs: list[dict] = list(palette_cfg.get("plugins", []))
        self._plugin_cache: dict[str, SimulatorPlugin] = {}

        # Active plugin
        self.active_plugin_id: Optional[str]           = None
        self.active_plugin:    Optional[SimulatorPlugin] = None
        self.plugin_state:     dict                    = {}   # {item: {var: val}}
        self.current_params:   dict                    = {}   # {name: float}

        # Current simulation output (last step result)
        self.last_trajectories: dict = {}   # {item: {var: array(T,)}}

        # ── Coevolution scheduler ─────────────────────────────────────────────
        self._scheduler: Optional[CoevolutionScheduler] = None

        # ── Fitness ───────────────────────────────────────────────────────────
        # A simple RMS-energy metric accumulated over the last frame.
        self._fitness_accum: float = 0.0
        self._fitness_count: int   = 0

    # ── Plugin selection ──────────────────────────────────────────────────────

    @property
    def plugin_list(self) -> list[dict]:
        """The raw plugin definition dicts from the palette config."""
        return self._plugin_defs

    def select_plugin(self, plugin_id: str) -> bool:
        """Load and activate a plugin by ID.  Resets all simulation state.

        Returns True on success, False if the ID is not in the palette.
        """
        if not any(p["id"] == plugin_id for p in self._plugin_defs):
            return False

        # Load from cache or disk
        if plugin_id not in self._plugin_cache:
            try:
                self._plugin_cache[plugin_id] = load_plugin(plugin_id)
            except (FileNotFoundError, Exception):
                return False

        self.active_plugin_id = plugin_id
        self.active_plugin    = self._plugin_cache[plugin_id]

        # Initialise default params from PARAM_SPECS
        self.current_params = {
            spec["name"]: float(spec.get("default", spec.get("min", 0.0)))
            for spec in self.active_plugin.param_specs
        }

        # Initialise plugin state
        self.plugin_state     = self.active_plugin.make_state(self.n_items)
        self.last_trajectories = {}

        # Reset coevolution scheduler with new param specs
        self._scheduler = CoevolutionScheduler(
            self._coevo_cfg,
            param_specs=self.active_plugin.param_specs,
        )
        self._fitness_accum = 0.0
        self._fitness_count = 0

        self.mode = SimulatorMode.IDLE
        return True

    # ── Transport controls ────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin normal (non-evolving) simulation."""
        if self.active_plugin is None:
            return
        self.mode = SimulatorMode.RUNNING

    def pause(self) -> None:
        if self.mode in (SimulatorMode.RUNNING, SimulatorMode.EVOLVING):
            self.mode = SimulatorMode.PAUSED

    def resume(self) -> None:
        if self.mode == SimulatorMode.PAUSED:
            self.mode = SimulatorMode.RUNNING

    def stop(self) -> None:
        self.mode = SimulatorMode.IDLE

    def start_evolving(self) -> None:
        """Switch to coevolution mode.  Restarts the scheduler if done."""
        if self.active_plugin is None:
            return
        if self._scheduler is None or self._scheduler.is_done:
            self._build_scheduler()
        self.mode = SimulatorMode.EVOLVING

    def stop_evolving(self) -> None:
        """Return to normal RUNNING mode, keeping the best-found params."""
        if self._scheduler is not None:
            stats = self._scheduler.get_stats()
            if stats["best_params"]:
                self.current_params.update(stats["best_params"])
        self.mode = SimulatorMode.RUNNING

    # ── Per-frame step ────────────────────────────────────────────────────────

    def step(
        self,
        inputs:  dict,
        dt:      float,
        n_items: Optional[int] = None,
    ) -> dict:
        """Run one block step of the active plugin.

        Call this from the physics thread once per frame when in RUNNING or
        EVOLVING mode.

        Parameters
        ----------
        inputs:
            Signal routing dict ``{src_key: array(T,)}``.
        dt:
            Seconds per sample.
        n_items:
            Overrides ``self.n_items`` when provided.

        Returns
        -------
        dict
            Raw output from ``plugin.step()``.
        """
        if self.active_plugin is None:
            return {}
        if self.mode not in (SimulatorMode.RUNNING, SimulatorMode.EVOLVING):
            return {}

        n = n_items if n_items is not None else self.n_items

        # Pick parameter source
        if self.mode == SimulatorMode.EVOLVING and self._scheduler is not None:
            params = self._scheduler.active_params
        else:
            params = self.current_params

        result = self.active_plugin.step(
            inputs, self.plugin_state, dt, n_items=n,
            use_torch=self.use_torch, params=params,
        )

        # Extract trajectories and updated state
        if isinstance(result, dict) and "outputs" in result:
            self.last_trajectories = result.get("outputs", {})
            if "state" in result:
                self.plugin_state = result["state"]
        else:
            self.last_trajectories = result
            # Module-level plugins don't return updated state; keep as-is.

        # Fitness accumulation for coevolution
        if self.mode == SimulatorMode.EVOLVING:
            self._accumulate_fitness()
            if self._scheduler is not None:
                self._scheduler.tick()

        return self.last_trajectories

    # ── Parameter editing ─────────────────────────────────────────────────────

    def set_param(self, name: str, value: float) -> None:
        """Directly set a single parameter (while IDLE or PAUSED)."""
        if self.active_plugin is None:
            return
        self.current_params[name] = float(value)

    # ── Evolution stats ───────────────────────────────────────────────────────

    @property
    def population_stats(self) -> dict:
        """Latest coevolution scheduler stats (empty dict if not active)."""
        if self._scheduler is None:
            return {}
        return self._scheduler.get_stats()

    # ── Introspection helpers ─────────────────────────────────────────────────

    @property
    def param_specs(self) -> list[dict]:
        if self.active_plugin is None:
            return []
        return self.active_plugin.param_specs

    @property
    def output_vars(self) -> list[str]:
        if self.active_plugin is None:
            return []
        return self.active_plugin.effective_output_vars

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_scheduler(self) -> None:
        specs = self.active_plugin.param_specs if self.active_plugin else []
        self._scheduler = CoevolutionScheduler(self._coevo_cfg, param_specs=specs)

    def _accumulate_fitness(self) -> None:
        """Compute an RMS-energy fitness score from last_trajectories."""
        total = 0.0
        count = 0
        for item_traj in self.last_trajectories.values():
            if not isinstance(item_traj, dict):
                continue
            for arr in item_traj.values():
                if isinstance(arr, np.ndarray) and arr.size:
                    total += float(np.mean(arr ** 2))
                    count += 1
        if count:
            rms = math.sqrt(total / count)
            if self._scheduler is not None:
                self._scheduler.push_fitness(rms)
            self._fitness_accum += rms
            self._fitness_count += 1

    def __repr__(self) -> str:
        return (f"SimulatorWorkspace(mode={self.mode.value!r}, "
                f"plugin={self.active_plugin_id!r})")
