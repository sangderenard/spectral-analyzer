"""camera_designer/neural_assembly.py
=====================================
Bidirectional surface-to-surface noodle manifold MLP.

MLP input (5) — entry-surface canonical frame (angles relative to theta_hit):
    r_in          radial hit distance on entry surface (m)
    dir_r_in      radial direction component
    dir_phi_in    azimuthal direction component (relative to theta_hit)
    dir_z_in      axial direction (positive = into surface)
    wavelength_um vacuum wavelength (µm)

MLP output (6) — all angles relative to theta_hit:
    r_out         radial distance on exit surface (m)
    delta_phi     exit azimuth offset = theta_out − theta_hit (radians)
    dir_r_out     radial direction component at exit
    dir_phi_out   azimuthal direction component at exit
    dir_z_out     axial direction at exit (positive = away from entry)
    opl           optical path length entry→exit (m)

Backward transport uses the same trained weights with z_entry/z_exit swapped in the
payload header (side=1).  The canonical frame at the back surface is identical in
format to the front — no separate training pass required.
"""
from __future__ import annotations

import math
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

__all__ = [
    "NeuralAssemblyMLP", "NormStats",
    "train", "export_payload", "load_training_data",
    "infer_payload", "fit_acceptance_boundary",
]

MAGIC_NEURAL  = 14948.0
# Noodle table: one row per ray that made it through the assembly.
# Inputs (5): r_in, dir_r_in, dir_phi_in, dir_z_in, wavelength_um
#   — all expressed in entry-surface canonical frame (angles relative to theta_hit)
#   — dir_z_in stored as positive (entering-surface-normal-aligned)
# Outputs (6): r_out, delta_phi, dir_r_out, dir_phi_out, dir_z_out, opl
#   — all angles relative to theta_hit of the entry surface
N_INPUTS      = 5
N_OUTPUTS     = 6
HEADER_FLOATS = 16
NORM_FLOATS   = N_INPUTS * 2 + N_OUTPUTS * 2   # 22
LAYER_OFFSET  = HEADER_FLOATS + NORM_FLOATS     # 38

# Training table column indices (11-col schema from bake_neural_training_data)
_T_R_IN       = 0
_T_DIR_R_IN   = 1
_T_DIR_PHI_IN = 2
_T_DIR_Z_IN   = 3
_T_WL         = 4
_T_R_OUT      = 5
_T_DELTA_PHI  = 6
_T_DIR_R_OUT  = 7
_T_DIR_PHI_OUT= 8
_T_DIR_Z_OUT  = 9
_T_OPL        = 10


# ─────────────────────────────────────────────────────────────────────────────
# Normalization statistics
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """Input/output normalization constants derived from training data.

    Input columns 0-5 are normalized by mean and std across all rows.

    Output columns:
      throughput (col 6):  binary; no normalization — sigmoid applied in network
      sensor_x/y (cols 7,8): mean/std computed over hit-only rows
      dir_x/y/z (cols 9-11): unit vectors — no normalization (mean=0, scale=1)
      opl (col 12):          mean/std computed over hit-only rows
    """

    def __init__(self,
                 in_mean:   np.ndarray,
                 in_scale:  np.ndarray,
                 out_mean:  np.ndarray,
                 out_scale: np.ndarray,
                 ) -> None:
        self.in_mean   = np.asarray(in_mean,   np.float32)
        self.in_scale  = np.asarray(in_scale,  np.float32)
        self.out_mean  = np.asarray(out_mean,  np.float32)
        self.out_scale = np.asarray(out_scale, np.float32)

    @classmethod
    def from_data(cls, data: np.ndarray) -> "NormStats":
        """Compute normalization stats from an (N, 11) noodle table."""
        in_mean  = np.mean(data[:, :N_INPUTS], axis=0).astype(np.float32)
        in_std   = np.std(data[:, :N_INPUTS],  axis=0).astype(np.float32)
        in_scale = np.where(in_std > 1e-8, in_std,
                            np.ones(N_INPUTS, np.float32)).astype(np.float32)

        out_cols = data[:, N_INPUTS:N_INPUTS + N_OUTPUTS]
        out_mean  = np.mean(out_cols, axis=0).astype(np.float32)
        out_std   = np.std(out_cols,  axis=0).astype(np.float32)
        # Angles (delta_phi, dir_phi_in, dir_phi_out) have O(1) std — don't over-scale
        out_scale = np.where(out_std > 1e-8, out_std,
                             np.ones(N_OUTPUTS, np.float32)).astype(np.float32)
        # Angles are already O(1); clamp their scale to [0.1, 10] to avoid tiny corrections
        for i in (1, 3, 4):  # delta_phi, dir_phi_out, dir_phi_out indices in output
            out_scale[i] = float(np.clip(out_scale[i], 0.1, 10.0))

        return cls(in_mean, in_scale, out_mean, out_scale)

    def normalize_inputs(self, x: "torch.Tensor") -> "torch.Tensor":
        m = torch.as_tensor(self.in_mean,  dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.in_scale, dtype=x.dtype, device=x.device)
        return (x - m) / s

    def normalize_outputs_inplace(self, y: "torch.Tensor") -> "torch.Tensor":
        """Normalize all outputs by (val - mean) / scale."""
        om  = torch.as_tensor(self.out_mean,  dtype=y.dtype, device=y.device)
        os_ = torch.as_tensor(self.out_scale, dtype=y.dtype, device=y.device)
        return (y - om) / os_


# ─────────────────────────────────────────────────────────────────────────────
# Parametric acceptance boundary
# ─────────────────────────────────────────────────────────────────────────────

def fit_acceptance_boundary(
    r_in:    np.ndarray,
    dir_z_in: np.ndarray,
    alive:   np.ndarray,
    r_lens:  float,
    n_bins:  int   = 16,
    margin:  float = 0.04,
) -> tuple[float, float]:
    """Fit quadratic screen: cos_theta_min(r) = c0 + c1*(r/r_lens)^2.

    Returns (c0, c1).  T2 rejects ray if dir_z_in < c0 + c1*(r_in/r_lens)^2.
    Returns (0.0, 0.0) when insufficient data to determine a boundary.

    The boundary is shared between the LUT pre-filter and MLP inference — it
    describes the same acceptance cone that would bound a full grid LUT.
    """
    r_max = max(float(r_lens), 1e-9)
    rn    = np.asarray(r_in,    np.float64) / r_max
    dz    = np.asarray(dir_z_in, np.float64)
    ok    = np.asarray(alive,   bool)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ctr: list[float] = []
    bin_thr: list[float] = []
    for i in range(n_bins):
        mask  = (rn >= edges[i]) & (rn < edges[i + 1])
        trans = dz[mask & ok]
        if len(trans) < 8:
            continue
        # 5th percentile of transmitted cos_theta ≈ acceptance lower bound
        thr = float(np.percentile(trans, 5)) - margin
        bin_ctr.append((edges[i] + edges[i + 1]) * 0.5)
        bin_thr.append(thr)

    if len(bin_thr) < 3:
        return 0.0, 0.0

    rn2    = np.array(bin_ctr, np.float64) ** 2
    A      = np.column_stack([np.ones_like(rn2), rn2])
    coeffs, _, _, _ = np.linalg.lstsq(A, np.array(bin_thr, np.float64), rcond=None)
    c0 = max(0.0, float(coeffs[0]))
    c1 = float(coeffs[1])
    return c0, c1


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class NeuralAssemblyMLP(nn.Module):
    """N_INPUTS(5) → [hidden]*n_hidden → N_OUTPUTS(6) MLP.

    Returns normalized outputs; caller denormalizes with NormStats.
    """

    def __init__(self, hidden_dim: int = 256, n_hidden: int = 4) -> None:
        if not _TORCH_OK:
            raise ImportError("torch is required for NeuralAssemblyMLP")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hidden   = n_hidden

        layers: list[nn.Module] = []
        in_d = N_INPUTS
        for _ in range(n_hidden):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, N_OUTPUTS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(
    path: str,
    max_rows: Optional[int] = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """Load (N, 11) float32 table from bake_neural_training_data."""
    raw = np.load(path, mmap_mode="r")
    if max_rows is not None and len(raw) > max_rows:
        idx = np.random.default_rng(rng_seed).choice(
            len(raw), max_rows, replace=False)
        idx.sort()
        raw = raw[idx]
    return np.asarray(raw, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _angular_l1(pred: "torch.Tensor", tgt: "torch.Tensor") -> "torch.Tensor":
    """Mean L1 angular error, wrapping correctly through ±π."""
    diff = pred - tgt
    # wrap to [-π, π]
    diff = diff - (2 * math.pi) * torch.round(diff / (2 * math.pi))
    return diff.abs().mean()


def _compute_loss(
    raw_out: "torch.Tensor",   # (B, 6) raw network output (normalized)
    target:  "torch.Tensor",   # (B, 6) normalized targets
) -> "torch.Tensor":
    """Huber loss on radii/opl, angular L1 on phi outputs.
    Outputs: [r_out, delta_phi, dir_r_out, dir_phi_out, dir_z_out, opl]
    """
    r_loss    = F.huber_loss(raw_out[:, 0], target[:, 0], delta=1.0)
    dphi_loss = _angular_l1( raw_out[:, 1], target[:, 1])
    dr_loss   = F.l1_loss(   raw_out[:, 2], target[:, 2])
    phi_loss  = _angular_l1( raw_out[:, 3], target[:, 3])
    dz_loss   = F.l1_loss(   raw_out[:, 4], target[:, 4])
    opl_loss  = F.huber_loss(raw_out[:, 5], target[:, 5], delta=1.0)
    return r_loss + dphi_loss + dr_loss + phi_loss + dz_loss + opl_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _run_training_loop(X, Y_norm, model, epochs, batch_size, lr, verbose, tag):
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    N = len(X)
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx].to(model.net[0].weight.device), Y_norm[idx].to(model.net[0].weight.device)
            opt.zero_grad()
            loss = _compute_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches  += 1
        sched.step()
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"[{tag}] epoch {epoch+1:3d}/{epochs}  "
                  f"loss={epoch_loss/max(n_batches,1):.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
    model.eval()


def train(
    data_path:  str,
    epochs:     int   = 60,
    batch_size: int   = 8192,
    lr:         float = 1e-3,
    hidden_dim: int   = 256,
    n_hidden:   int   = 4,
    max_rows:   Optional[int] = None,
    device:     Optional[str] = None,
    verbose:    bool  = True,
) -> tuple["NeuralAssemblyMLP", NormStats]:
    """Train forward (front→back) MLP from an 11-col noodle table."""
    if not _TORCH_OK:
        raise ImportError("torch is required for training")
    dev  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = load_training_data(data_path, max_rows=max_rows)
    if verbose:
        print(f"[neural_assembly.train] {len(data):,} noodles  device={dev}", flush=True)
    norm  = NormStats.from_data(data)
    X     = torch.tensor((data[:, :N_INPUTS] - norm.in_mean) / norm.in_scale, dtype=torch.float32)
    Y_raw = torch.tensor(data[:, N_INPUTS:N_INPUTS + N_OUTPUTS], dtype=torch.float32)
    Y_norm = norm.normalize_outputs_inplace(Y_raw)
    model = NeuralAssemblyMLP(hidden_dim=hidden_dim, n_hidden=n_hidden).to(dev)
    _run_training_loop(X, Y_norm, model, epochs, batch_size, lr, verbose, "neural_assembly.train")
    return model, norm


def train_from_array(
    data:       np.ndarray,
    epochs:     int   = 60,
    batch_size: int   = 8192,
    lr:         float = 1e-3,
    hidden_dim: int   = 256,
    n_hidden:   int   = 4,
    max_rows:   Optional[int] = None,
    device:     Optional[str] = None,
    verbose:    bool  = True,
) -> tuple["NeuralAssemblyMLP", NormStats]:
    """Train forward MLP from an in-memory (N, 11) float32 noodle array.

    Identical to train() but takes a numpy array instead of a file path,
    so the progressive bake loop can train without touching the filesystem.
    """
    if not _TORCH_OK:
        raise ImportError("torch is required for training")
    dev  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = np.asarray(data, dtype=np.float32)
    if max_rows is not None and len(data) > max_rows:
        idx = np.random.default_rng(0).choice(len(data), max_rows, replace=False)
        idx.sort()
        data = data[idx]
    if verbose:
        print(f"[neural_assembly.train_from_array] {len(data):,} rows  device={dev}",
              flush=True)
    norm  = NormStats.from_data(data)
    X     = torch.tensor((data[:, :N_INPUTS] - norm.in_mean) / norm.in_scale, dtype=torch.float32)
    Y_raw = torch.tensor(data[:, N_INPUTS:N_INPUTS + N_OUTPUTS], dtype=torch.float32)
    Y_norm = norm.normalize_outputs_inplace(Y_raw)
    model = NeuralAssemblyMLP(hidden_dim=hidden_dim, n_hidden=n_hidden).to(dev)
    _run_training_loop(X, Y_norm, model, epochs, batch_size, lr, verbose,
                       "neural_assembly.train_from_array")
    return model, norm




# ─────────────────────────────────────────────────────────────────────────────
# Payload export
# ─────────────────────────────────────────────────────────────────────────────

def export_payload(
    model:       "NeuralAssemblyMLP",
    norm:        NormStats,
    z_entry:     float,         # this surface plane (entry)
    z_exit:      float,         # destination surface plane (exit)
    r_lens:      float = 0.0,   # physical lens radius for hit check (0 = skip)
    boundary_c0: float = 0.0,   # acceptance screen constant term
    boundary_c1: float = 0.0,   # acceptance screen quadratic coeff (r/r_lens)^2
) -> np.ndarray:
    """Serialize trained model to a magic-14948 float32 payload array.

    Register with::

        tracer.add_scale_context(
            ...,
            context_kind = _sk.SCALE_CONTEXT_KIND_NEURAL_SURFACE,
            payload      = export_payload(...),
        )
    """
    if not _TORCH_OK:
        raise ImportError("torch is required for export_payload")

    hidden_dim = model.hidden_dim
    n_hidden   = model.n_hidden
    n_layers   = n_hidden + 1

    layers_wb: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for module in model.net:
            if isinstance(module, nn.Linear):
                W = module.weight.cpu().numpy().astype(np.float32)
                b = module.bias.cpu().numpy().astype(np.float32)
                layers_wb.append((W, b))

    if len(layers_wb) != n_layers:
        raise ValueError(f"Expected {n_layers} Linear layers, found {len(layers_wb)}")

    # ── Header (HEADER_FLOATS = 16) ──────────────────────────────────────────
    header = np.array([
        MAGIC_NEURAL,          # [0]
        float(n_layers),       # [1]
        float(N_INPUTS),       # [2]  5
        float(hidden_dim),     # [3]
        float(N_OUTPUTS),      # [4]  6
        float(z_entry),        # [5]
        float(z_exit),         # [6]
        float(boundary_c0),    # [7]  acceptance screen constant (c0)
        float(boundary_c1),    # [8]  acceptance screen quadratic coeff (c1)
        0.0,                   # [9] reserved
        float(r_lens),         # [10]
        0.0, 0.0, 0.0, 0.0, 0.0,  # [11..15] ROC, k, axis_idx, r_out, side (set by registration)
    ], dtype=np.float32)
    assert len(header) == HEADER_FLOATS

    # ── Normalization block (NORM_FLOATS = 22) ───────────────────────────────
    # in_mean[5], in_scale[5], out_mean[6], out_scale[6]
    norm_block = np.concatenate([
        norm.in_mean.astype(np.float32),   # [16..20]
        norm.in_scale.astype(np.float32),  # [21..25]
        norm.out_mean.astype(np.float32),  # [26..31]
        norm.out_scale.astype(np.float32), # [32..37]
    ])
    assert len(norm_block) == NORM_FLOATS

    # ── Layer weight blocks (from index 38 onward) ───────────────────────────
    layer_blocks: list[np.ndarray] = []
    for W, b in layers_wb:
        layer_blocks.append(W.ravel())
        layer_blocks.append(b)

    return np.concatenate([header, norm_block] + layer_blocks).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Python-side inference (mirrors apply_neural_mlp_from_f32 in ray_tracer.cpp)
# ─────────────────────────────────────────────────────────────────────────────

def infer_payload(
    payload: np.ndarray,
    x_hit:   float,  # Cartesian hit position on entry surface
    y_hit:   float,
    dir_x:   float,  # incoming ray direction (unit vector)
    dir_y:   float,
    dir_z:   float,
    wl:      float,  # wavelength in microns
) -> dict:
    """Run one inference pass mirroring the C++ apply_neural_mlp_from_f32.

    Converts Cartesian hit + direction to entry-surface canonical frame,
    runs the MLP, and returns exit position + direction in Cartesian.

    Returns dict: {exit_x, exit_y, exit_z, out_dir (unit vec), opl}
    """
    p = payload
    assert p[0] == MAGIC_NEURAL, f"wrong magic: {p[0]}"

    n_layers   = int(p[1])
    input_dim  = int(p[2])
    hidden_dim = int(p[3])
    output_dim = int(p[4])
    z_exit_val = float(p[6])

    assert input_dim  == N_INPUTS,  f"input_dim mismatch: {input_dim}"
    assert output_dim == N_OUTPUTS, f"output_dim mismatch: {output_dim}"

    in_mean  = p[HEADER_FLOATS                        : HEADER_FLOATS + N_INPUTS]
    in_scale = p[HEADER_FLOATS + N_INPUTS             : HEADER_FLOATS + 2*N_INPUTS]
    out_mean = p[HEADER_FLOATS + 2*N_INPUTS           : HEADER_FLOATS + 2*N_INPUTS + N_OUTPUTS]
    out_scl  = p[HEADER_FLOATS + 2*N_INPUTS + N_OUTPUTS : LAYER_OFFSET]

    # ── Entry-surface canonical frame ────────────────────────────────────────
    theta_hit = math.atan2(y_hit, x_hit)
    r_hit     = math.hypot(x_hit, y_hit)
    cos_t     = math.cos(theta_hit)
    sin_t     = math.sin(theta_hit)
    dir_r_in  =  dir_x * cos_t + dir_y * sin_t
    dir_phi_in= -dir_x * sin_t + dir_y * cos_t
    dir_z_in  = abs(dir_z)  # always positive (toward surface normal)

    # Parametric acceptance boundary check (same screen as T2 GLSL)
    bnd_c0   = float(p[7])
    bnd_c1   = float(p[8])
    r_lens_v = float(p[10])
    if (bnd_c0 != 0.0 or bnd_c1 != 0.0) and r_lens_v > 0.0:
        r_norm = r_hit / r_lens_v
        if dir_z_in < bnd_c0 + bnd_c1 * r_norm * r_norm:
            return {"rejected": True, "exit_x": 0.0, "exit_y": 0.0,
                    "exit_z": z_exit_val, "out_dir": np.array([0., 0., 1.]), "opl": 0.0}

    raw_in = np.array([r_hit, dir_r_in, dir_phi_in, dir_z_in, wl], np.float32)
    x = (raw_in - in_mean) / np.maximum(np.abs(in_scale), 1e-12)

    # ── Forward pass ─────────────────────────────────────────────────────────
    ptr = LAYER_OFFSET
    for l in range(n_layers):
        in_d  = input_dim  if l == 0           else hidden_dim
        out_d = output_dim if l == n_layers - 1 else hidden_dim
        W = p[ptr : ptr + out_d * in_d].reshape(out_d, in_d)
        b = p[ptr + out_d * in_d : ptr + out_d * in_d + out_d]
        ptr += out_d * in_d + out_d
        x = W @ x + b
        if l < n_layers - 1:
            x = np.maximum(x, 0.0)

    # ── Denormalize ──────────────────────────────────────────────────────────
    raw_out = x * out_scl + out_mean

    r_out       = float(raw_out[_T_R_OUT - N_INPUTS])
    delta_phi   = float(raw_out[_T_DELTA_PHI - N_INPUTS])
    dir_r_out   = float(raw_out[_T_DIR_R_OUT - N_INPUTS])
    dir_phi_out = float(raw_out[_T_DIR_PHI_OUT - N_INPUTS])
    dir_z_out   = float(raw_out[_T_DIR_Z_OUT - N_INPUTS])
    opl         = float(raw_out[_T_OPL - N_INPUTS])

    # ── Reconstruct Cartesian exit ───────────────────────────────────────────
    theta_out = theta_hit + delta_phi
    exit_x = r_out * math.cos(theta_out)
    exit_y = r_out * math.sin(theta_out)
    # direction: dir_phi_out is still relative to theta_hit frame
    d_x = dir_r_out * cos_t - dir_phi_out * sin_t
    d_y = dir_r_out * sin_t + dir_phi_out * cos_t
    d   = np.array([d_x, d_y, dir_z_out], np.float64)
    d  /= max(float(np.linalg.norm(d)), 1e-12)

    return {
        "exit_x":  exit_x,
        "exit_y":  exit_y,
        "exit_z":  z_exit_val,
        "out_dir": d,
        "opl":     opl,
    }
