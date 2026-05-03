"""sm_plugins/_base.py
======================
SimulatorPlugin — class-based plugin interface for the SimulatorStation.

Provides two things:

1.  A base class ``SimulatorPlugin`` with a clean ``make_state / step`` API
    that new plugins can subclass.

2.  ``wrap_module(mod)`` — a factory that wraps any existing module-level
    sm_plugin (``STATE_VARS`` / ``step()`` convention) into the class-based
    interface so that all existing plugins in this package are automatically
    compatible with the SimulatorStation without any code changes.

3.  ``load_plugin(plugin_id)`` — convenience loader that imports a plugin
    module by name and returns a ``SimulatorPlugin`` instance.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorPlugin:
    """Abstract base for simulator plugins.

    Subclass this to write a new class-based plugin, **or** rely on
    ``wrap_module`` to adapt any existing module-level plugin automatically.
    """

    #: Scalar state variable names persisted between steps.
    state_vars:  list[str] = []
    #: Variable names that become routing output nodes.  Falls back to
    #: ``state_vars`` when omitted.
    output_vars: list[str] = []
    #: UI knob specs ``[{name, min, max, default, label, ...}]``
    param_specs: list[dict] = []
    #: Prefix for item names produced by ``make_state``; default ``"m"``.
    item_prefix: str = "m"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def make_state(self, n_items: int = 1) -> dict:
        """Return an initial state dict ``{item_name: {var: 0.0, …}}``."""
        return {
            f"{self.item_prefix}{i}": {v: 0.0 for v in self.state_vars}
            for i in range(n_items)
        }

    def step(
        self,
        inputs: dict,
        state: dict,
        dt: float,
        n_items: int = 1,
        use_torch: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Run one block step.

        Parameters
        ----------
        inputs:
            ``{src_key: array(T,)}`` of routed signals.
        state:
            ``{item_name: {var_name: float}}`` scalar state from last step.
        dt:
            Seconds per sample.
        n_items:
            Number of independent items to simulate.
        use_torch:
            Backend preference flag.
        **kwargs:
            Plugin-specific keyword arguments (e.g. ``params={...}``).

        Returns
        -------
        dict
            Trajectories ``{item_name: {var_name: array(T,)}}`` — or the
            extended form ``{"outputs": …, "state": …, "plugin_state": …}``.
        """
        raise NotImplementedError

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def effective_output_vars(self) -> list[str]:
        return self.output_vars if self.output_vars else self.state_vars

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"state={self.state_vars!r}, "
                f"params={[p.get('name') for p in self.param_specs]!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level plugin adaptor
# ─────────────────────────────────────────────────────────────────────────────

class _ModuleWrapper(SimulatorPlugin):
    """Wraps a module-level sm_plugin (STATE_VARS / step) as SimulatorPlugin."""

    def __init__(self, mod: Any) -> None:
        self._mod = mod
        self.state_vars  = list(getattr(mod, "STATE_VARS",  []))
        self.output_vars = list(getattr(mod, "OUTPUT_VARS", self.state_vars))
        self.param_specs = list(getattr(mod, "PARAM_SPECS", []))
        self.item_prefix = str(getattr(mod, "ITEM_PREFIX", "m"))
        name = getattr(mod, "__name__", "") or ""
        self.name: str = name.split(".")[-1]

    def step(self, inputs, state, dt, n_items=1, use_torch=False, **kwargs):
        return self._mod.step(
            inputs, state, dt,
            n_items=n_items, use_torch=use_torch,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"_ModuleWrapper({self.name!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Public factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_module(mod: Any) -> SimulatorPlugin:
    """Wrap an existing sm_plugins module into the class-based interface."""
    return _ModuleWrapper(mod)


def load_plugin(plugin_id: str, plugins_dir: str | None = None) -> SimulatorPlugin:
    """Load and return a plugin by ID.

    Searches ``sm_plugins/`` by default (the same directory as this file).
    If the plugin module exports a ``SimulatorPlugin`` subclass that class is
    instantiated and returned; otherwise the module is wrapped via
    :func:`wrap_module`.

    Raises
    ------
    FileNotFoundError
        If ``<plugins_dir>/<plugin_id>.py`` does not exist.
    """
    if plugins_dir is None:
        plugins_dir = os.path.dirname(__file__)

    path = os.path.join(plugins_dir, f"{plugin_id}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sm_plugin not found: {path}")

    spec = importlib.util.spec_from_file_location(
        f"sm_plugins.{plugin_id}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    # Prefer a native SimulatorPlugin subclass declared in the module.
    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and issubclass(attr, SimulatorPlugin)
            and attr is not SimulatorPlugin
        ):
            return attr()

    # Fall back to wrapping the module-level API.
    return wrap_module(mod)
