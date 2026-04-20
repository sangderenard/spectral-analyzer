"""
Template state-machine plugin.

Copy this file, rename it, then select the filename stem in the
AnalyticModule state-machine plugin dropdown.
"""

from __future__ import annotations

import numpy as np

STATE_VARS = ["value"]
OUTPUT_VARS = ["value"]
ITEM_PREFIX = "cell"
PARAM_SPECS = [
    {"name": "gain", "label": "Gain", "dtype": "float",
     "default": 1.0, "low": 0.0, "high": 4.0, "fmt": ".3f", "group": "Plugin"},
]


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def step(inputs, state, dt, n_items=1, use_torch=False, **kwargs):
    """Return {item_name: {var_name: array(T,)}} for one render buffer."""
    params = dict(kwargs.get("params", {}) or {})
    item_names = [f"{ITEM_PREFIX}{i}" for i in range(max(1, int(n_items)))]

    T = 0
    drive = None
    for arr in inputs.values():
        a = _to_numpy(arr)
        T = max(T, len(a))
        sig = a.real if np.iscomplexobj(a) else np.asarray(a, dtype=np.float64)
        drive = sig if drive is None else drive + sig
    if T <= 0:
        T = 1
    if drive is None:
        drive = np.zeros(T, dtype=np.float64)
    else:
        drive = np.asarray(drive[:T], dtype=np.float64)

    out = {}
    for item_name in item_names:
        x = float(state.get(item_name, {}).get("value", 0.0))
        traj = np.empty(T, dtype=np.float64)
        for i in range(T):
            x += dt * (-0.5 * x + float(params.get("gain", 1.0)) * drive[i])
            traj[i] = x
        out[item_name] = {"value": traj}
    return out
