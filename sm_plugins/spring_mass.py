"""
State machine plugin: N coupled spring-mass oscillators in 2D.

ITEMS  — one entry per mass: "m0", "m1", ...  (generated from n_items at runtime)
STATE_VARS — per-item state variables

step() receives:
    inputs   : {src_key: array(T,) complex128}  — any routing signals wired in
    state    : {item: {var: float}}              — initial conditions
    dt       : float                             — seconds per sample
    n_items  : int
    use_torch: bool

Returns:
    trajectories: {item: {var: array(T,) float64}}
"""

import numpy as np

STATE_VARS = ["pos_x", "pos_y", "vel_x", "vel_y"]

# Spring / physics constants (tunable by overriding from inputs keyed "k", "damp")
_K_DEFAULT   = 20.0    # spring stiffness  (rad/s)^2
_DAMP_DEFAULT = 0.05   # damping coefficient


def step(inputs, state, dt, n_items=1, use_torch=False, **kwargs):
    """
    Integrate spring-mass system for one buffer (T samples).

    Any input signal keyed to a source node is summed as a drive force
    split equally across all items in the x-direction.
    """
    # Resolve to numpy regardless of torch flag
    def _np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)

    items = [f"m{i}" for i in range(n_items)]

    # Build initial-condition arrays
    px = np.array([float(state.get(it, {}).get("pos_x", 0.0)) for it in items])
    py = np.array([float(state.get(it, {}).get("pos_y", 0.0)) for it in items])
    vx = np.array([float(state.get(it, {}).get("vel_x", 0.0)) for it in items])
    vy = np.array([float(state.get(it, {}).get("vel_y", 0.0)) for it in items])

    # Aggregate drive force from all inputs (real part → x, imag part → y)
    T = 0
    drive_x = drive_y = None
    for arr in inputs.values():
        a = _np(arr)
        if T == 0:
            T = len(a)
        else:
            T = max(T, len(a))
        fx = a.real if np.iscomplexobj(a) else a
        fy = a.imag if np.iscomplexobj(a) else np.zeros_like(a)
        drive_x = fx if drive_x is None else drive_x + fx
        drive_y = fy if drive_y is None else drive_y + fy

    if T == 0:
        T = 1
    if drive_x is None:
        drive_x = np.zeros(T)
        drive_y = np.zeros(T)

    drive_x = np.asarray(drive_x[:T], dtype=np.float64)
    drive_y = np.asarray(drive_y[:T], dtype=np.float64)

    k    = _K_DEFAULT
    damp = _DAMP_DEFAULT

    # Output trajectory buffers: (n_items, T)
    out_px = np.empty((n_items, T))
    out_py = np.empty((n_items, T))
    out_vx = np.empty((n_items, T))
    out_vy = np.empty((n_items, T))

    # Symplectic Euler integration (sample by sample)
    for t in range(T):
        fx = -k * px + drive_x[t] / max(n_items, 1)
        fy = -k * py + drive_y[t] / max(n_items, 1)
        # Damping
        fx -= damp * vx
        fy -= damp * vy
        # Nearest-neighbour coupling between items
        if n_items > 1:
            px_shifted = np.roll(px, 1)
            px_shifted[0] = px[-1]
            fx += k * 0.25 * (px_shifted - px)
        vx = vx + dt * fx
        vy = vy + dt * fy
        px = px + dt * vx
        py = py + dt * vy
        out_px[:, t] = px
        out_py[:, t] = py
        out_vx[:, t] = vx
        out_vy[:, t] = vy

    # Build return dict {item: {var: array(T,)}}
    traj = {}
    for i, it in enumerate(items):
        traj[it] = {
            "pos_x": out_px[i],
            "pos_y": out_py[i],
            "vel_x": out_vx[i],
            "vel_y": out_vy[i],
        }
    return traj
