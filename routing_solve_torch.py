"""routing_solve_torch.py — Torch port of the analytic routing solver.

Faithful complex128 / float64 port of every function in ``routing_engine.py``
that touches signal data.  The graph metadata (RoutingGraph, RoutingEdge, etc.)
stays in the original module — this module only replaces the compute path.

Precision contract
------------------
* All signal tensors are **torch.complex128** (backed by float64 real+imag).
* All scalar parameter series are **torch.float64**.
* No dtype downcasting anywhere.  No approximations.  If the numpy path does
  it, the torch path does the same arithmetic in the same precision.

Device contract
---------------
Every public function accepts a ``device`` argument (or infers it from input
tensors).  The same code runs on CPU and CUDA with no branches.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

# Re-use the graph dataclasses — no need to redefine them.
from routing_engine import (
    MetaEdge,
    FeedbackConfig,
    ParamEdge,
    RoutingEdge,
    RoutingGraph,
    RouterInstance,
)

_CDTYPE = torch.complex128
_FDTYPE = torch.float64

def _meta_endpoint_key(graph: RoutingGraph, owner_key: str, port_key: str) -> str:
    """Pick a concrete graph node for one metaedge endpoint.

    Authoring may target bundle ports that are not themselves registered graph
    nodes yet.  The compiled router still needs a node key, so prefer the owner
    node when present and fall back to the port key only when necessary.
    """
    graph_nodes = set(graph.node_keys())
    if owner_key and owner_key in graph_nodes:
        return owner_key
    if port_key and port_key in graph_nodes:
        return port_key
    return owner_key or port_key


def lower_meta_edges(graph: RoutingGraph) -> list[RoutingEdge]:
    """Lower authored MetaEdge objects into directed RoutingEdges.

    This preserves graph ideology at authoring time while giving the current
    torch router a concrete directed-edge approximation it can compile today.
    """
    lowered: list[RoutingEdge] = []
    for me in getattr(graph, "meta_edges", []):
        if not isinstance(me, MetaEdge):
            continue
        a_key = _meta_endpoint_key(graph, me.a_key, me.a_port)
        b_key = _meta_endpoint_key(graph, me.b_key, me.b_port)
        if not a_key or not b_key:
            continue
        lowered.append(RoutingEdge(
            src_key=a_key,
            dst_key=b_key,
            weight=1.0,
            angle_rad=0.0,
            delay_s=float(getattr(me, "delay_s", 0.0)),
            edge_kind="meta",
            src_port=me.a_port,
            dst_port=me.b_port,
            tensor_contract=me.tensor_contract,
            transfer_spec=me.a_to_b_transfer,
        ))
        lowered.append(RoutingEdge(
            src_key=b_key,
            dst_key=a_key,
            weight=1.0,
            angle_rad=0.0,
            delay_s=float(getattr(me, "delay_s", 0.0)),
            edge_kind="meta",
            src_port=me.b_port,
            dst_port=me.a_port,
            tensor_contract=me.tensor_contract,
            transfer_spec=me.b_to_a_transfer,
        ))
    return lowered


def project_tensor_to_scalar_param(
    x: Tensor,
    policy: str = "magnitude_mean",
    time_dim: int | None = None,
) -> Tensor:
    """Project a complex tensor onto scalar parameter state values.

    This is a receiver-side policy, not a transport policy.  The input tensor
    may carry batch or instance structure.  When ``time_dim`` is omitted, the
    projector preserves leading dimensions and collapses only the final
    lane/channel dimension when the selected policy requires it.  When
    ``time_dim`` is provided, every non-time dimension is treated as tensor
    multiplicity and collapsed according to the policy while the time axis is
    preserved.
    """
    if x.ndim == 0:
        x = x.reshape(1)
    p = str(policy or "magnitude_mean")
    if time_dim is None:
        reduce_dims = (-1,) if x.ndim > 1 else ()
        lane_axis = x.ndim - 1
    else:
        td = int(time_dim)
        if td < 0:
            td += x.ndim
        reduce_dims = tuple(i for i in range(x.ndim) if i != td)
        lane_axis = next((i for i in range(x.ndim) if i != td), td)

    def _reduce(y: Tensor, mode: str) -> Tensor:
        if not reduce_dims:
            return y
        if mode == "sum":
            return y.sum(dim=reduce_dims)
        return y.mean(dim=reduce_dims)

    if p == "real":
        return x.real.to(_FDTYPE)
    if p == "imag":
        return x.imag.to(_FDTYPE)
    if p == "magnitude":
        return x.abs().to(_FDTYPE)
    if p == "phase":
        return torch.angle(x).to(_FDTYPE)
    if p == "real_mean":
        return _reduce(x.real.to(_FDTYPE), "mean")
    if p == "real_sum":
        return _reduce(x.real.to(_FDTYPE), "sum")
    if p == "imag_mean":
        return _reduce(x.imag.to(_FDTYPE), "mean")
    if p == "imag_sum":
        return _reduce(x.imag.to(_FDTYPE), "sum")
    if p == "magnitude_sum":
        return _reduce(x.abs().to(_FDTYPE), "sum")
    if p == "phase_mean":
        return _reduce(torch.angle(x).to(_FDTYPE), "mean")
    if p == "lane0_real":
        return x.real.to(_FDTYPE).select(lane_axis, 0)
    if p == "lane0_imag":
        return x.imag.to(_FDTYPE).select(lane_axis, 0)
    if p == "lane0_magnitude":
        return x.abs().to(_FDTYPE).select(lane_axis, 0)
    return _reduce(x.abs().to(_FDTYPE), "mean")


# ═══════════════════════════════════════════════════════════════════════════════
# Soft-clip (overflow guard)
# ═══════════════════════════════════════════════════════════════════════════════

_SOFTCLIP_KNEE: float = 1e200


def _softclip_complex(X: Tensor) -> Tensor:
    """In-place magnitude-preserving soft-clip for complex128 tensors.

    Identity for |z| < knee (~1e200).  Tanh-compresses magnitude while
    preserving phase exactly for the astronomically rare overflow case.
    """
    mag = X.abs()
    mask = mag >= _SOFTCLIP_KNEE
    if not mask.any():
        return X
    m_in = mag[mask]
    m_out = _SOFTCLIP_KNEE * torch.tanh(m_in / _SOFTCLIP_KNEE)
    scale = torch.ones_like(m_in)
    nz = m_in > 0
    scale[nz] = m_out[nz] / m_in[nz]
    X[mask] = X[mask] * scale.to(_CDTYPE)
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Safe inverse
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_inverse(mat: Tensor) -> Tensor:
    """Complex128 matrix inverse with jitter fallback."""
    try:
        return torch.linalg.inv(mat)
    except Exception:
        jitter = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device) * 1e-6
        return torch.linalg.inv(mat + jitter)


# ═══════════════════════════════════════════════════════════════════════════════
# Weakly connected components (CPU-side graph partitioning)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_wccs(
    node_keys: list[str],
    edges: list[RoutingEdge],
    coupled_pairs: list[list[str]],
) -> list[list[int]]:
    """Union-find WCC partitioning — identical logic to numpy version."""
    n = len(node_keys)
    if n == 0:
        return []
    ki = {k: i for i, k in enumerate(node_keys)}
    parent = list(range(n))

    def _find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for e in edges:
        si, di = ki.get(e.src_key), ki.get(e.dst_key)
        if si is not None and di is not None:
            _union(si, di)
    for pair in coupled_pairs:
        idxs = [ki[k] for k in pair if k in ki]
        for j in range(1, len(idxs)):
            _union(idxs[0], idxs[j])

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = _find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Core solver
# ═══════════════════════════════════════════════════════════════════════════════

# Node transform type: receives a (T,) complex128 row and returns (T,) complex128.
NodeTransformFn = Callable[[Tensor], Tensor]
# Coupled transform: receives {key: (T,) complex128} → {key: (T,) complex128}.
CoupledTransformFn = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


def solve_routing_complex(
    Src: Tensor,                                      # (N, T) complex128
    edges: list[RoutingEdge],
    node_keys: list[str],
    sr: float,
    global_decay: float = 1.0,
    node_transforms: Optional[Dict[str, NodeTransformFn]] = None,
    coupled_transforms: Optional[Dict[Tuple[str, ...], CoupledTransformFn]] = None,
    max_iterations: int = 1000,
    convergence_eps: float = 1e-12,
) -> Tensor:                                          # (N, T) complex128
    """Solve the analytic routing system in torch complex128.

    Exact port of ``routing_engine.solve_routing_complex``.

    Signal model per edge::

        x_dst[t] += weight * exp(i * angle_rad) * x_src[t − delay_samples]

    For instantaneous (delay=0) linear graphs: ``X = (I − W)⁻¹ @ Src``.

    For graphs with node transforms or coupled transforms: iterative
    fixed-point to convergence (or max_iterations).

    For delayed edges: causal chunk-by-chunk processing with the
    instantaneous system pre-inverted.

    No approximations.  No precision changes.  complex128 throughout.
    """
    ki = {k: i for i, k in enumerate(node_keys)}
    N, T = Src.shape
    dev = Src.device

    # ── WCC partitioning ──────────────────────────────────────────────
    _coupled_pairs = [list(kt) for kt in (coupled_transforms or {})]
    _components = _find_wccs(node_keys, edges, _coupled_pairs)

    if len(_components) > 1:
        _transform_keys = set(node_transforms or {})
        _coupled_keys: set[str] = set()
        for _kt in (coupled_transforms or {}):
            _coupled_keys.update(_kt)

        X = Src.clone()
        _work: list[list[int]] = []
        for _comp in _components:
            if len(_comp) == 1:
                _key = node_keys[_comp[0]]
                if _key not in _transform_keys and _key not in _coupled_keys:
                    continue
            _work.append(_comp)

        for _comp_indices in _work:
            _comp_set = {node_keys[i] for i in _comp_indices}
            _sub_keys = [node_keys[i] for i in _comp_indices]
            _idx_t = torch.tensor(_comp_indices, dtype=torch.long, device=dev)
            _sub_Src = Src[_idx_t]
            _sub_edges = [
                e for e in edges
                if e.src_key in _comp_set and e.dst_key in _comp_set
            ]
            _sub_nt = (
                {k: v for k, v in (node_transforms or {}).items() if k in _comp_set}
                or None
            )
            _sub_ct = (
                {kt: fn for kt, fn in (coupled_transforms or {}).items()
                 if all(k in _comp_set for k in kt)}
                or None
            )
            _result = solve_routing_complex(
                _sub_Src, _sub_edges, _sub_keys, sr, global_decay,
                _sub_nt, _sub_ct, max_iterations, convergence_eps,
            )
            X[_idx_t] = _result
        return X

    # ── Single component ──────────────────────────────────────────────
    transforms = node_transforms or {}

    transform_idx: set[int] = set()
    for key in transforms:
        ti = ki.get(key)
        if ti is not None:
            transform_idx.add(ti)

    coupled_groups: list[tuple[tuple[str, ...], list[int], CoupledTransformFn]] = []
    for keys_tuple, fn in (coupled_transforms or {}).items():
        idxs = [ki[k] for k in keys_tuple if k in ki]
        if len(idxs) == len(keys_tuple):
            coupled_groups.append((keys_tuple, idxs, fn))

    # ── Build weight matrices ─────────────────────────────────────────
    W_inst = torch.zeros(N, N, dtype=_CDTYPE, device=dev)
    delay_groups: dict[int, Tensor] = {}
    adv_groups: dict[int, Tensor] = {}

    for e in edges:
        si, di = ki.get(e.src_key), ki.get(e.dst_key)
        if si is None or di is None:
            continue
        cw = complex(
            e.weight * global_decay * math.cos(e.angle_rad),
            e.weight * global_decay * math.sin(e.angle_rad),
        )
        d_samp = int(round(e.delay_s * sr))
        if d_samp == 0:
            W_inst[di, si] += cw
        elif d_samp > 0:
            if d_samp not in delay_groups:
                delay_groups[d_samp] = torch.zeros(N, N, dtype=_CDTYPE, device=dev)
            delay_groups[d_samp][di, si] += cw
        else:
            adv = -d_samp
            if adv not in adv_groups:
                adv_groups[adv] = torch.zeros(N, N, dtype=_CDTYPE, device=dev)
            adv_groups[adv][di, si] += cw

    # Fold pre-advance (negative-delay) contributions into Src
    if adv_groups:
        Src = Src.clone()
        for adv, Wadv in adv_groups.items():
            if adv < T:
                Src[:, :T - adv] += Wadv @ Src[:, adv:]

    # Pre-invert instantaneous linear part
    IW = torch.eye(N, dtype=_CDTYPE, device=dev) - W_inst
    M = _safe_inverse(IW)

    has_transforms = len(transform_idx) > 0 or len(coupled_groups) > 0

    # ── Transform helpers ─────────────────────────────────────────────
    def _apply_transforms(X: Tensor) -> Tensor:
        for key, fn in transforms.items():
            ti = ki.get(key)
            if ti is not None:
                X[ti] = fn(X[ti])
        for keys_tuple, idxs, fn in coupled_groups:
            rows_in = {k: X[i] for k, i in zip(keys_tuple, idxs)}
            rows_out = fn(rows_in)
            for k, i in zip(keys_tuple, idxs):
                if k in rows_out:
                    X[i] = rows_out[k]
        return X

    # ── Partition into linear / nonlinear ─────────────────────────────
    nl_idx = sorted(transform_idx | {i for _, idxs, _ in coupled_groups for i in idxs})
    li_idx = [i for i in range(N) if i not in nl_idx]

    if has_transforms and nl_idx and li_idx:
        nl_t = torch.tensor(nl_idx, dtype=torch.long, device=dev)
        li_t = torch.tensor(li_idx, dtype=torch.long, device=dev)

        W_ll = W_inst[li_t][:, li_t]
        W_ln = W_inst[li_t][:, nl_t]
        W_nl = W_inst[nl_t][:, li_t]
        W_nn = W_inst[nl_t][:, nl_t]
        Ainv = _safe_inverse(
            torch.eye(len(li_idx), dtype=_CDTYPE, device=dev) - W_ll
        )
        rhs_n_const = W_nl @ Ainv
        W_eff = W_nn + W_nl @ Ainv @ W_ln

        local_of_global = {gi: i for i, gi in enumerate(nl_idx)}
        reduced_transform_idx = [local_of_global[i] for i in sorted(transform_idx)]
        reduced_single_transforms: dict[int, NodeTransformFn] = {
            local_of_global[gi]: transforms[node_keys[gi]]
            for gi in sorted(transform_idx)
        }
        reduced_coupled_groups: list[tuple[tuple[str, ...], list[int], CoupledTransformFn]] = []
        for keys_tuple, idxs, fn in coupled_groups:
            reduced_coupled_groups.append(
                (keys_tuple, [local_of_global[i] for i in idxs], fn)
            )

        def _apply_reduced_transforms(X_n: Tensor) -> Tensor:
            for ti in reduced_transform_idx:
                X_n[ti] = reduced_single_transforms[ti](X_n[ti])
            for keys_tuple, idxs, fn in reduced_coupled_groups:
                rows_in = {k: X_n[i] for k, i in zip(keys_tuple, idxs)}
                rows_out = fn(rows_in)
                for k, i in zip(keys_tuple, idxs):
                    if k in rows_out:
                        X_n[i] = rows_out[k]
            return X_n

        def _solve_reduced(rhs: Tensor) -> Tensor:
            rhs_l = rhs[li_t]
            rhs_n = rhs[nl_t]
            rhs_eff = rhs_n + rhs_n_const @ rhs_l
            M_eff = _safe_inverse(
                torch.eye(len(nl_idx), dtype=_CDTYPE, device=dev) - W_eff
            )
            X_n = M_eff @ rhs_eff
            _apply_reduced_transforms(X_n)
            _softclip_complex(X_n)
            for _ in range(max_iterations):
                X_prev = X_n.clone()
                X_input_n = rhs_eff + W_eff @ X_n
                X_n = X_input_n.clone()
                _apply_reduced_transforms(X_n)
                _softclip_complex(X_n)
                if (X_n - X_prev).abs().max() < convergence_eps:
                    break
            X_out = torch.zeros(N, rhs.shape[1], dtype=_CDTYPE, device=dev)
            X_out[nl_t] = X_n
            X_out[li_t] = Ainv @ (rhs_l + W_ln @ X_n)
            return X_out
    else:
        def _solve_reduced(rhs: Tensor) -> Tensor:
            X = M @ rhs
            _apply_transforms(X)
            _softclip_complex(X)
            for _ in range(max_iterations):
                X_prev = X.clone()
                X_input = rhs + W_inst @ X
                X = X_input.clone()
                for ti in transform_idx:
                    X[ti] = transforms[node_keys[ti]](X_input[ti])
                for keys_tuple, idxs, fn in coupled_groups:
                    rows_in = {k: X_input[i] for k, i in zip(keys_tuple, idxs)}
                    rows_out = fn(rows_in)
                    for k, i in zip(keys_tuple, idxs):
                        if k in rows_out:
                            X[i] = rows_out[k]
                _softclip_complex(X)
                if (X - X_prev).abs().max() < convergence_eps:
                    break
            return X

    # ── No delays, no transforms: exact linear solve ──────────────────
    if not delay_groups and not has_transforms:
        return M @ Src

    # ── No delays, with transforms: iterative fixed-point ─────────────
    if not delay_groups:
        return _solve_reduced(Src)

    # ── Delayed edges: causal chunk solver ─────────────────────────────
    d_min = min(delay_groups.keys())
    X = torch.zeros(N, T, dtype=_CDTYPE, device=dev)

    for t0 in range(0, T, d_min):
        t1 = min(t0 + d_min, T)

        rhs = Src[:, t0:t1].clone()
        for d, Wd in delay_groups.items():
            t_past = t0 - d
            if t_past >= 0:
                chunk_len = t1 - t0
                rhs += Wd @ X[:, t_past:t_past + chunk_len]

        if not has_transforms:
            X[:, t0:t1] = M @ rhs
        else:
            X[:, t0:t1] = _solve_reduced(rhs)

    return X


# ═══════════════════════════════════════════════════════════════════════════════
# Ringdown
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_ringdown_samples(
    edges: list[RoutingEdge],
    sr: float,
    global_decay: float,
    fb: FeedbackConfig,
) -> int:
    """Estimate extra samples for routing tail — identical to numpy version."""
    if fb.ringdown_mode == "none":
        return 0
    max_n = int(fb.ringdown_max_s * sr)
    if max_n <= 0:
        return 0
    if fb.ringdown_mode == "fixed":
        return max_n

    delayed = [
        (e.delay_s, abs(e.weight) * global_decay)
        for e in edges if e.delay_s > 1e-9 and abs(e.weight) > 1e-12
    ]
    if not delayed:
        return 0
    max_w = max(w for _, w in delayed)
    if max_w <= 0.0:
        return 0
    if max_w >= 1.0:
        return max_n

    max_d_samp = max(int(round(d * sr)) for d, _ in delayed)
    if max_d_samp <= 0:
        return 0

    threshold = max(fb.ringdown_threshold, 1e-12)
    bounces = math.ceil(math.log(threshold) / math.log(max_w))
    return min(max_n, bounces * max_d_samp)


def solve_routing_with_ringdown(
    Src: Tensor,                                      # (N, T) complex128
    edges: list[RoutingEdge],
    node_keys: list[str],
    sr: float,
    global_decay: float,
    fb: FeedbackConfig,
    node_transforms: Optional[Dict[str, NodeTransformFn]] = None,
    coupled_transforms: Optional[Dict[Tuple[str, ...], CoupledTransformFn]] = None,
) -> tuple[Tensor, int]:
    """Routing solve with zero-padded ringdown tail.

    Returns ``(X_full, note_len)`` where ``X_full.shape[1] >= T`` and
    ``note_len == T`` (the original signal length).
    """
    note_len = Src.shape[1]
    ringdown_n = estimate_ringdown_samples(edges, sr, global_decay, fb)
    if ringdown_n > 0:
        tail = torch.zeros(
            Src.shape[0], ringdown_n, dtype=_CDTYPE, device=Src.device,
        )
        Src_ext = torch.cat([Src, tail], dim=1)
    else:
        Src_ext = Src
    X_full = solve_routing_complex(
        Src_ext, edges, node_keys, sr, global_decay,
        node_transforms=node_transforms,
        coupled_transforms=coupled_transforms,
    )
    return X_full, note_len


# ═══════════════════════════════════════════════════════════════════════════════
# Latency compensation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latency_compensation(
    edges: list[RoutingEdge],
    node_keys: list[str],
    sr: float,
) -> dict[str, int]:
    """Pre-roll per source node — identical logic to numpy version.

    Returns {node_key: lead_samples} for nodes that need pre-roll.
    Pure graph algorithm — no tensor math, CPU-only.
    """
    ki = {k: i for i, k in enumerate(node_keys)}
    N = len(node_keys)

    direct_delay: dict[tuple[int, int], int] = {}
    for e in edges:
        si, di = ki.get(e.src_key), ki.get(e.dst_key)
        if si is None or di is None or e.delay_s <= 1e-9:
            continue
        d_samp = int(round(e.delay_s * sr))
        if d_samp <= 0:
            continue
        key_pair = (si, di)
        direct_delay[key_pair] = max(direct_delay.get(key_pair, 0), d_samp)

    if not direct_delay:
        return {}

    # Reachability via BFS (cycle detection)
    adj: list[list[int]] = [[] for _ in range(N)]
    for e in edges:
        si, di = ki.get(e.src_key), ki.get(e.dst_key)
        if si is not None and di is not None:
            adj[si].append(di)

    reachable = [[False] * N for _ in range(N)]
    for start in range(N):
        stack = list(adj[start])
        visited: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reachable[start][cur] = True
            stack.extend(adj[cur])

    # Feedforward edges only (exclude feedback)
    ff_edges = {
        (si, di): d for (si, di), d in direct_delay.items()
        if not reachable[di][si]
    }
    if not ff_edges:
        return {}

    # Longest-path via Bellman-Ford relaxation
    dist = [[0] * N for _ in range(N)]
    for (si, di), d in ff_edges.items():
        dist[si][di] = max(dist[si][di], d)

    for _ in range(N - 1):
        changed = False
        for (si, di), d in ff_edges.items():
            for j in range(N):
                if dist[di][j] > 0 or di == j:
                    new_d = d + (dist[di][j] if di != j else 0)
                    if new_d > dist[si][j]:
                        dist[si][j] = new_d
                        changed = True
        if not changed:
            break

    result: dict[str, int] = {}
    for i, key in enumerate(node_keys):
        lead = max(dist[i])
        if lead > 0:
            result[key] = lead
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Parametric extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_param_series(arr: Tensor, extractor: str) -> Tensor:
    """Convert complex128 time series to float64 scalar series.

    Exact port of all extractors from routing_engine.extract_param_series.
    """
    if extractor == "real":
        return arr.real
    if extractor == "imag":
        return arr.imag
    if extractor == "phase":
        return torch.angle(arr)
    if extractor == "energy":
        return arr.real ** 2 + arr.imag ** 2
    if extractor == "rms":
        mag = arr.abs()
        win = 128
        kernel = torch.ones(win, dtype=_FDTYPE, device=arr.device) / win
        # Convolve mag^2 with uniform kernel, then sqrt — same-mode
        mag_sq = mag ** 2
        # F.conv1d requires (batch, channel, length)
        mag_sq_3d = mag_sq.unsqueeze(0).unsqueeze(0).to(_FDTYPE)
        kernel_3d = kernel.unsqueeze(0).unsqueeze(0)
        # Pad for "same" mode
        pad = win // 2
        convolved = torch.nn.functional.conv1d(
            mag_sq_3d, kernel_3d, padding=pad,
        )
        # Trim to original length (conv1d with padding=win//2 can produce +1)
        result = convolved.squeeze()[:arr.shape[0]]
        return result.sqrt()
    # default: "magnitude"
    return arr.abs().to(_FDTYPE)


def solve_param_routing(
    signal_map: Dict[str, Tensor],            # node_key → complex128 (T,)
    param_edges: list[ParamEdge],
    param_defaults: Dict[str, float],
    param_bounds: Dict[str, tuple[float, float]],
) -> Dict[str, Tensor]:                       # param_node_key → float64 (T,)
    """Accumulate scalar parameter series from ParamEdge extraction rules.

    Exact port of routing_engine.solve_param_routing.
    """
    T = max((v.shape[-1] for v in signal_map.values()), default=0)
    if T == 0:
        return {
            key: torch.tensor([default], dtype=_FDTYPE)
            for key, default in param_defaults.items()
        }

    # Infer device from first signal
    dev = next(iter(signal_map.values())).device

    result: Dict[str, Tensor] = {
        key: torch.full((T,), default, dtype=_FDTYPE, device=dev)
        for key, default in param_defaults.items()
    }

    for e in param_edges:
        if e.src_key not in signal_map:
            continue
        src = signal_map[e.src_key]
        if src.ndim > 1:
            extracted = project_tensor_to_scalar_param(
                src,
                getattr(e, "projection_policy", "magnitude_mean"),
                time_dim=src.ndim - 1,
            ) * e.weight
        else:
            extracted = extract_param_series(src, e.extractor) * e.weight
        dst = e.dst_key
        if dst not in result:
            result[dst] = torch.zeros(T, dtype=_FDTYPE, device=dev)
        L = min(extracted.shape[0], result[dst].shape[0])
        result[dst][:L] += extracted[:L]

    for key, (lo, hi) in param_bounds.items():
        if key in result:
            result[key] = result[key].clamp(lo, hi)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LFO synthesis
# ═══════════════════════════════════════════════════════════════════════════════

def _lfo_waveform(
    shape: str,
    t: Tensor,
    rate_hz: float,
    phase_offset: float,
    depth: float,
) -> Tensor:
    """Generate real-valued LFO waveform.  (T,) float64."""
    ph = 2.0 * math.pi * rate_hz * t + phase_offset
    if shape == "Sine":
        return depth * torch.sin(ph)
    elif shape == "Triangle":
        return depth * (2.0 * (2.0 * (ph / (2.0 * math.pi) % 1.0) - 1.0).abs() - 1.0)
    elif shape == "Sawtooth":
        return depth * (2.0 * (ph / (2.0 * math.pi) % 1.0) - 1.0)
    else:  # Square
        return depth * torch.sign(torch.sin(ph))


def _hilbert_torch(x: Tensor) -> Tensor:
    """Compute the analytic signal via Hilbert transform in torch.

    Uses the frequency-domain approach: zero the negative frequencies,
    double the positive frequencies.  This is mathematically identical
    to ``scipy.signal.hilbert`` — no approximation.

    Input: (T,) real float64.  Output: (T,) complex128.
    """
    N = x.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=_CDTYPE, device=x.device)

    Xf = torch.fft.fft(x.to(_CDTYPE))

    # Build the h array: h[0]=1, h[N/2]=1 (if even), h[1..N/2-1]=2, h[N/2+1..]=0
    h = torch.zeros(N, dtype=_FDTYPE, device=x.device)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[1:(N + 1) // 2] = 2.0

    return torch.fft.ifft(Xf * h.to(_CDTYPE))


def synthesize_lfo_csig(
    rate_hz: float,
    shape: str,
    phase_offset: float,
    depth: float,
    n: int,
    sr: float,
    t_offset: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Synthesize an LFO as a complex128 analytic signal.

    Exact port of ``_synthesize_lfo_csig`` from analytic_driver.py.

    For Sine: exact analytic form ``depth * exp(i*(ωt + φ))``.
    For others: Hilbert transform of the real waveform.
    """
    t = (torch.arange(n, dtype=_FDTYPE, device=device) / max(sr, 1.0)) + t_offset

    if shape == "Sine":
        omega = 2.0 * math.pi * rate_hz
        return (depth * torch.exp(1j * (omega * t + phase_offset))).to(_CDTYPE)

    real = _lfo_waveform(shape, t, rate_hz, phase_offset, depth)
    return _hilbert_torch(real)


def synthesize_lfo_channel_csig(
    ch: dict,
    n: int,
    sr: float,
    t_offset: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Synthesize one LFO channel dict to complex128 analytic signal.

    Exact port of ``_synthesize_lfo_channel_csig`` from analytic_driver.py.
    Handles: rate_hz, amplitude, phase_offset, shape, tension, resample (ZOH),
    slew_order (1|2), slew (0-1).
    """
    rate_hz = float(ch.get("rate_hz", 1.0))
    amplitude = float(ch.get("amplitude", 1.0))
    phase_offset = float(ch.get("phase_offset", 0.0))
    shape = ch.get("shape", "Sine")
    tension = max(1e-3, float(ch.get("tension", 1.0)))
    resample = max(1, int(ch.get("resample", 1)))
    slew_order = max(1, min(2, int(ch.get("slew_order", 1))))
    slew_val = float(max(0.0, min(0.9999, float(ch.get("slew", 0.0)))))

    t = (torch.arange(n, dtype=_FDTYPE, device=device) / max(sr, 1.0)) + t_offset
    ph = 2.0 * math.pi * rate_hz * t + phase_offset

    # Raw waveform
    if shape == "Sine":
        raw = torch.sin(ph)
    elif shape == "Triangle":
        raw = 2.0 * (2.0 * (ph / (2.0 * math.pi) % 1.0) - 1.0).abs() - 1.0
    elif shape == "Sawtooth":
        raw = 2.0 * (ph / (2.0 * math.pi) % 1.0) - 1.0
    else:  # Square
        raw = torch.sign(torch.sin(ph))

    # Tension shaping: sign(x) * |x|^tension
    shaped = torch.sign(raw) * raw.abs().pow(tension)

    # Resample — zero-order hold
    if resample > 1:
        decimated = shaped[::resample]
        shaped = decimated.repeat_interleave(resample)[:n]
        if shaped.shape[0] < n:
            pad = shaped[-1:].expand(n - shaped.shape[0])
            shaped = torch.cat([shaped, pad])

    # Slew — exponential IIR low-pass (1st or 2nd order)
    # This is inherently sequential — must scan.
    if slew_val > 1e-6:
        alpha = max(1e-6, (1.0 - slew_val) ** 2)
        # First-order IIR: y[n] = y[n-1] + alpha*(x[n] - y[n-1])
        # = (1-alpha)*y[n-1] + alpha*x[n]
        # Use cumulative scan for efficiency when possible, but for exact
        # match with the numpy version (which uses scipy.signal.lfilter with
        # b=[alpha], a=[1, -(1-alpha)]), we implement the same IIR.
        for _pass in range(slew_order):
            out = torch.empty_like(shaped)
            y = shaped[0].item()
            for k in range(n):
                y = y + alpha * (shaped[k].item() - y)
                out[k] = y
            shaped = out

    # Hilbert lift to analytic signal
    return amplitude * _hilbert_torch(shaped)


# ═══════════════════════════════════════════════════════════════════════════════
# Envelope — fully parametric piecewise polynomial evaluation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   1. Solve for polynomial coefficients once (K knots → (K-1) segments × 4 coeffs).
#      This is O(K) — trivially small for any envelope.
#   2. Map each of N output samples to its segment index — one searchsorted.
#   3. Evaluate  a·dx³ + b·dx² + c·dx + d  in pure vector arithmetic.
#      No interpolation.  The polynomial *is* the function at every sample.
#
# Three coefficient solvers:
#   - _linear_coeffs:  piecewise linear (a=0, b=0, c=slope, d=y_i)
#   - _clamped_spline_coeffs:  clamped cubic spline S'(endpoints)=0
#   - _pchip_coeffs:  monotone cubic Hermite (Fritsch–Carlson)


def compute_envelope(
    knot_times: Tensor,    # (K,) float64 — absolute times (s)
    knot_values: Tensor,   # (K,) float64 — envelope values
    n: int,
    duration: float,
    env_type: str = "adsr",
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Evaluate an envelope at n exact sample positions.  (n,) float64.

    All modes produce a ``(K-1, 4)`` coefficient matrix, then evaluate
    every sample via vector polynomial arithmetic.  No interpolation —
    the polynomial resolves exactly at the sample rate.
    """
    K = knot_times.shape[0]
    # endpoint=False — n samples in [0, duration), matching numpy path
    t_ax = torch.arange(n, dtype=_FDTYPE, device=device) * (duration / max(n, 1))

    kt = knot_times.to(device)
    kv = knot_values.to(device)

    if env_type == "spline" and K >= 4:
        coeffs = _clamped_spline_coeffs(kt, kv)
    elif env_type == "monotone" and K >= 2:
        coeffs = _pchip_coeffs(kt, kv)
    else:
        coeffs = _linear_coeffs(kt, kv)

    return _eval_piecewise_poly(kt, coeffs, t_ax).clamp(min=0.0)


# ── Segment lookup + vectorised polynomial evaluation ─────────────────────

def _eval_piecewise_poly(
    xs: Tensor,      # (K,) knot positions
    coeffs: Tensor,  # (K-1, 4) — [a, b, c, d] per segment
    t: Tensor,       # (N,) query positions
) -> Tensor:
    """Evaluate piecewise cubic at every query point.

    For segment i, the polynomial is::

        S_i(t) = a_i·dx³ + b_i·dx² + c_i·dx + d_i
        dx = t − x_i

    Pure vector math — no Python loop over samples.
    """
    K = xs.shape[0]
    seg = torch.searchsorted(xs.contiguous(), t.contiguous()).sub_(1).clamp_(0, K - 2)
    dx = t - xs[seg]
    a = coeffs[seg, 0]
    b = coeffs[seg, 1]
    c = coeffs[seg, 2]
    d = coeffs[seg, 3]
    # Horner form: ((a·dx + b)·dx + c)·dx + d
    return ((a * dx + b) * dx + c) * dx + d


# ── Piecewise linear coefficients ────────────────────────────────────────

def _linear_coeffs(xs: Tensor, ys: Tensor) -> Tensor:
    """(K-1, 4) coefficients for piecewise-linear segments.

    a = 0, b = 0, c = slope, d = y_i.
    """
    h = (xs[1:] - xs[:-1]).clamp(min=1e-30)
    slope = (ys[1:] - ys[:-1]) / h
    K1 = h.shape[0]
    dev = xs.device
    coeffs = torch.zeros(K1, 4, dtype=_FDTYPE, device=dev)
    coeffs[:, 2] = slope   # c
    coeffs[:, 3] = ys[:-1] # d
    return coeffs


# ── Clamped cubic spline coefficients ────────────────────────────────────

def _clamped_spline_coeffs(xs: Tensor, ys: Tensor) -> Tensor:
    """Solve for clamped cubic spline coefficients — S'(endpoints)=0.

    Returns (K-1, 4) tensor of [a, b, c, d] per segment, where::

        S_i(t) = a_i·dx³ + b_i·dx² + c_i·dx + d_i,  dx = t − x_i

    The tridiagonal solve runs on K unknowns (tiny).  All output is
    vector arithmetic.
    """
    K = xs.shape[0]
    dev = xs.device

    if K < 3:
        return _linear_coeffs(xs, ys)

    h = xs[1:] - xs[:-1]                               # (K-1,)
    delta = (ys[1:] - ys[:-1]) / h.clamp(min=1e-30)    # (K-1,)

    # ── K×K tridiagonal system for second derivatives m_i ─────────────
    #
    # Row 0  (clamped left,  S'(x_0) = 0):
    #     2h₀·m₀ + h₀·m₁ = 6δ₀
    # Row i  (interior, 1 ≤ i ≤ K-2):
    #     h_{i-1}·m_{i-1} + 2(h_{i-1}+h_i)·m_i + h_i·m_{i+1} = 6(δ_i − δ_{i-1})
    # Row K-1  (clamped right, S'(x_{K-1}) = 0):
    #     h_{K-2}·m_{K-2} + 2h_{K-2}·m_{K-1} = −6δ_{K-2}

    lower = torch.zeros(K, dtype=_FDTYPE, device=dev)
    diag  = torch.zeros(K, dtype=_FDTYPE, device=dev)
    upper = torch.zeros(K, dtype=_FDTYPE, device=dev)
    rhs   = torch.zeros(K, dtype=_FDTYPE, device=dev)

    # Row 0
    diag[0]  = 2.0 * h[0]
    upper[0] = h[0]
    rhs[0]   = 6.0 * delta[0]

    # Interior rows (vectorised)
    interior = torch.arange(1, K - 1, device=dev)
    lower[interior] = h[interior - 1]
    diag[interior]  = 2.0 * (h[interior - 1] + h[interior])
    upper[interior] = h[interior]
    rhs[interior]   = 6.0 * (delta[interior] - delta[interior - 1])

    # Row K-1
    lower[K - 1] = h[K - 2]
    diag[K - 1]  = 2.0 * h[K - 2]
    rhs[K - 1]   = -6.0 * delta[K - 2]

    m = _thomas_solve(lower, diag, upper, rhs)      # (K,) second derivatives

    # ── Build (K-1, 4) coefficient matrix ─────────────────────────────
    #  a_i = (m_{i+1} − m_i) / (6 h_i)
    #  b_i = m_i / 2
    #  c_i = δ_i − h_i (2 m_i + m_{i+1}) / 6
    #  d_i = y_i
    coeffs = torch.empty(K - 1, 4, dtype=_FDTYPE, device=dev)
    hi = h.clamp(min=1e-30)
    coeffs[:, 0] = (m[1:] - m[:-1]) / (6.0 * hi)
    coeffs[:, 1] = m[:-1] / 2.0
    coeffs[:, 2] = delta - hi * (2.0 * m[:-1] + m[1:]) / 6.0
    coeffs[:, 3] = ys[:-1]
    return coeffs


# ── PCHIP coefficients (Fritsch–Carlson monotone Hermite) ────────────────

def _pchip_coeffs(xs: Tensor, ys: Tensor) -> Tensor:
    """Monotone cubic Hermite coefficients — Fritsch–Carlson algorithm.

    Returns (K-1, 4) tensor of [a, b, c, d] per segment.  The Hermite
    form S_i(t) = d_i + c_i·dx + b_i·dx² + a_i·dx³ is converted from
    the standard basis {h00,h10,h01,h11} into power form analytically.
    """
    K = xs.shape[0]
    dev = xs.device

    if K < 2:
        return torch.zeros(max(K - 1, 0), 4, dtype=_FDTYPE, device=dev)

    h = xs[1:] - xs[:-1]                               # (K-1,)
    delta = (ys[1:] - ys[:-1]) / h.clamp(min=1e-30)    # (K-1,)

    # ── Derivatives at every knot (vectorised) ────────────────────────
    dk = torch.zeros(K, dtype=_FDTYPE, device=dev)

    if K == 2:
        dk[0] = delta[0]
        dk[1] = delta[0]
    else:
        # Interior: weighted harmonic mean where slopes agree in sign
        d1 = delta[:-1]     # δ_{k-1}
        d2 = delta[1:]      # δ_k
        h1 = h[:-1]         # h_{k-1}
        h2 = h[1:]          # h_k
        w1 = 2.0 * h2 + h1
        w2 = h2 + 2.0 * h1
        same = (d1.sign() == d2.sign()) & (d1.abs() > 1e-30) & (d2.abs() > 1e-30)
        denom = torch.where(same, w1 / d1 + w2 / d2, torch.ones_like(w1))
        dk[1:-1] = torch.where(same, (w1 + w2) / denom, torch.zeros_like(w1))

        # Endpoints (scipy _edge_case)
        dk[0]  = _pchip_edge(h[0], h[1], delta[0], delta[1])
        dk[-1] = _pchip_edge(h[-1], h[-2], delta[-1], delta[-2])

    # ── Convert Hermite {y_i, dk_i, y_{i+1}, dk_{i+1}} to power form ─
    #
    # The cubic Hermite in normalised coordinate s = dx/h is:
    #   P(s) = (2s³−3s²+1)·y_i + (s³−2s²+s)·h·dk_i
    #        + (−2s³+3s²)·y_{i+1} + (s³−s²)·h·dk_{i+1}
    #
    # Substituting dx = s·h, so s = dx/h, and expanding in dx:
    #   a = ( 2(y_i−y_{i+1}) + h(dk_i+dk_{i+1}) ) / h³
    #   b = ( 3(y_{i+1}−y_i) − h(2dk_i+dk_{i+1}) ) / h²
    #   c = dk_i
    #   d = y_i

    hi = h.clamp(min=1e-30)
    y0 = ys[:-1]
    y1 = ys[1:]
    d0 = dk[:-1]
    d1_k = dk[1:]

    coeffs = torch.empty(K - 1, 4, dtype=_FDTYPE, device=dev)
    coeffs[:, 0] = (2.0 * (y0 - y1) + hi * (d0 + d1_k)) / hi ** 3
    coeffs[:, 1] = (3.0 * (y1 - y0) - hi * (2.0 * d0 + d1_k)) / hi ** 2
    coeffs[:, 2] = d0
    coeffs[:, 3] = y0
    return coeffs


def _pchip_edge(h0: Tensor, h1: Tensor, m0: Tensor, m1: Tensor) -> Tensor:
    """One-sided shape-preserving endpoint derivative — scipy _edge_case."""
    val = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    cond1 = val.sign() != m0.sign()
    cond2 = (m0.sign() != m1.sign()) & (val.abs() > 3.0 * m0.abs())
    val = torch.where(cond1, torch.zeros_like(val), val)
    val = torch.where(~cond1 & cond2, 3.0 * m0, val)
    return val


# ── Thomas algorithm for tridiagonal systems ─────────────────────────────

def _thomas_solve(
    lower: Tensor, diag: Tensor, upper: Tensor, rhs: Tensor,
) -> Tensor:
    """Thomas algorithm — O(K) sequential on the knot count, not sample count.

    This is inherently sequential (each row depends on the previous), but K
    is the number of envelope knots (typically 4–20), not the sample count.
    Float64, exact, no pivoting needed for diagonally dominant systems.
    """
    K = diag.shape[0]
    if K == 0:
        return torch.zeros(0, dtype=_FDTYPE, device=diag.device)

    c = upper.clone()
    d = rhs.clone()
    b = diag.clone()

    # Forward elimination
    for i in range(1, K):
        w = lower[i] / b[i - 1].clamp(min=1e-30)
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Back substitution
    x = torch.zeros(K, dtype=_FDTYPE, device=diag.device)
    x[K - 1] = d[K - 1] / b[K - 1].clamp(min=1e-30)
    for i in range(K - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i].clamp(min=1e-30)

    return x


# ═══════════════════════════════════════════════════════════════════════════════
# High-level mixer (torch port)
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingMixerTorch:
    """Torch equivalent of routing_engine.RoutingMixer.

    Same API but operates on torch tensors.  All signal processing in
    complex128 on the specified device.
    """

    def __init__(
        self,
        graph: RoutingGraph,
        sample_rate: float = 48_000.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._graph = graph
        self._sr = float(sample_rate)
        self._device = device

    @property
    def graph(self) -> RoutingGraph:
        return self._graph

    @property
    def sample_rate(self) -> float:
        return self._sr

    @property
    def device(self) -> torch.device:
        return self._device

    def mix(
        self,
        sources: Dict[str, Tensor],
        node_transforms: Optional[Dict[str, NodeTransformFn]] = None,
        coupled_transforms: Optional[Dict[Tuple[str, ...], CoupledTransformFn]] = None,
    ) -> Dict[str, Tensor]:
        """Apply routing graph to named source tensors.

        Returns {node_key: complex128 (T,)} for every graph node.
        """
        keys = list(self._graph.nodes)
        for k in sources:
            if k not in keys:
                keys.append(k)

        T = max((v.shape[0] for v in sources.values()), default=0)
        if T == 0:
            return {k: torch.zeros(0, dtype=_CDTYPE, device=self._device) for k in keys}

        N = len(keys)
        Src = torch.zeros(N, T, dtype=_CDTYPE, device=self._device)
        for i, k in enumerate(keys):
            if k in sources:
                arr = sources[k].to(dtype=_CDTYPE, device=self._device)
                L = min(arr.shape[0], T)
                Src[i, :L] = arr[:L]

        g = self._graph
        global_decay = (
            max(0.0, 1.0 - float(g.feedback.decay))
            if g.feedback.enabled else 1.0
        )

        X = solve_routing_complex(
            Src, g.edges, keys, self._sr, global_decay,
            node_transforms=node_transforms,
            coupled_transforms=coupled_transforms,
        )
        return {k: X[i] for i, k in enumerate(keys)}

    def mix_to_mono(
        self,
        sources: Dict[str, Tensor],
        node_transforms: Optional[Dict[str, NodeTransformFn]] = None,
        coupled_transforms: Optional[Dict[Tuple[str, ...], CoupledTransformFn]] = None,
    ) -> Tensor:
        """Route sources and return element-wise sum of all node outputs."""
        out = self.mix(sources, node_transforms, coupled_transforms)
        if not out:
            return torch.zeros(0, dtype=_CDTYPE, device=self._device)
        arrays = list(out.values())
        total = arrays[0].clone()
        for a in arrays[1:]:
            total = total + a
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# Edge saturation functions
# ═══════════════════════════════════════════════════════════════════════════════

def _sat_tanh(z: Tensor, knee: float) -> Tensor:
    """Smooth magnetic-style saturation.  tanh(|z|/knee)·knee preserves phase."""
    mag = z.abs().clamp(min=1e-30)
    m_sat = knee * torch.tanh(mag / knee)
    return z * (m_sat / mag)


def _sat_hardclip(z: Tensor, knee: float) -> Tensor:
    """Hard DAC-style clip at *knee* amplitude.  Phase preserved below knee."""
    mag = z.abs().clamp(min=1e-30)
    m_sat = torch.clamp(mag, max=knee)
    return z * (m_sat / mag)


def _sat_softclip(z: Tensor, knee: float) -> Tensor:
    """Soft-clip above *knee* — tanh compression, identity below."""
    mag = z.abs()
    mask = mag >= knee
    if not mask.any():
        return z
    out = z.clone()
    m_in = mag[mask]
    m_out = knee * torch.tanh(m_in / knee)
    out[mask] = z[mask] * (m_out / m_in)
    return out


def _make_sat_fn(saturation: str, knee: float):
    """Return a saturation callable for the given type string, or None."""
    s = saturation.lower()
    if s == "tanh":
        return lambda z, k=knee: _sat_tanh(z, k)
    if s == "hardclip":
        return lambda z, k=knee: _sat_hardclip(z, k)
    if s in ("softclip", "soft"):
        return lambda z, k=knee: _sat_softclip(z, k)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-delay cycle detection
# ═══════════════════════════════════════════════════════════════════════════════

def find_zero_delay_cycles(
    node_keys: list[str],
    edges: "list[RoutingEdge]",
    sr: float,
) -> list[list[str]]:
    """Return all non-trivial SCCs in the zero-delay subgraph.

    Each returned list is a set of node keys that form a feedback cycle.
    An empty return value means the zero-delay graph is a DAG (exact solve
    in one pass; iteration is never needed for the linear case).

    Uses Tarjan's algorithm — O(V + E).
    """
    ki = {k: i for i, k in enumerate(node_keys)}
    n = len(node_keys)
    adj: list[list[int]] = [[] for _ in range(n)]
    for e in edges:
        si, di = ki.get(e.src_key), ki.get(e.dst_key)
        if si is None or di is None:
            continue
        if int(round(e.delay_s * sr)) == 0:
            adj[si].append(di)

    index_counter = [0]
    stack: list[int] = []
    lowlink: list[int] = [-1] * n
    index:   list[int] = [-1] * n
    on_stack: list[bool] = [False] * n
    sccs: list[list[str]] = []

    def _strongconnect(v: int) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if index[w] == -1:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(node_keys[w])
                if w == v:
                    break
            if len(scc) > 1 or (len(scc) == 1 and v in adj[v]):
                sccs.append(scc)

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 1000))
    for v in range(n):
        if index[v] == -1:
            _strongconnect(v)
    return sccs


# ═══════════════════════════════════════════════════════════════════════════════
# CompiledRouter — stateful per-sample solver
# ═══════════════════════════════════════════════════════════════════════════════

class CompiledRouter:
    """Stateful per-sample analytic signal router.

    Compile once from node_keys + edges, call step() once per sample.

    Linear solve
    ------------
    M = (I − W_lin)⁻¹ is pre-computed from all zero-delay edges that carry
    no saturation function.  Each sample: X = (src + delayed) @ M.T — one
    batched matmul, exact for the linear subgraph.

    Saturating edges
    ----------------
    Edges with a saturation type ("tanh", "hardclip", "softclip") are
    excluded from W_lin and handled by an iterative fixed-point solve after
    the linear solve.  The linear X serves as the initial guess, so the
    iteration needs very few steps for well-behaved signals.

    Cycle behaviour
    ---------------
    • Infinity flag — if any node magnitude exceeds *infinity_threshold*
      during iteration, ``last_saturated`` is set True and the iteration
      stops.  The saturation functions themselves bound the values, so the
      returned X is always finite.
    • Convergence — iteration stops when max|ΔX| < *convergence_eps*.
      For linear edges this is reached in exactly one step.  For saturating
      cycles it converges in a handful of steps; the saturation knee
      naturally limits loop gain and accelerates convergence.

    Batching
    --------
    src of shape (B, N) runs B independent instances through the same M in
    one matmul.  Ring buffers are (B, depth, N); every instance has its own
    causal delay state.
    """

    def __init__(
        self,
        node_keys: list[str],
        edges: "list[RoutingEdge]",
        sr: float,
        device: torch.device,
        global_decay: float = 1.0,
        batch_size: int = 1,
        max_iterations: int = 64,
        convergence_eps: float = 1e-10,
        infinity_threshold: float = 1e6,
    ) -> None:
        self.node_keys: list[str] = list(node_keys)
        self.N: int = len(node_keys)
        self.ki: dict[str, int] = {k: i for i, k in enumerate(node_keys)}
        self.device = device
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps
        self.infinity_threshold = infinity_threshold

        # Diagnostics updated each step()
        self.last_saturated: bool = False
        self.last_convergence_iters: int = 0

        W_lin = torch.zeros(self.N, self.N, dtype=_CDTYPE, device=device)
        delay_weights: dict[int, Tensor] = {}

        # Saturating instantaneous edges — handled iteratively
        # Each entry: (dst_idx, src_idx, complex_weight_tensor, sat_fn)
        self._sat_edges: list[tuple[int, int, Tensor, object]] = []

        for e in edges:
            si = self.ki.get(e.src_key)
            di = self.ki.get(e.dst_key)
            if si is None or di is None:
                continue
            cw = complex(
                e.weight * global_decay * math.cos(e.angle_rad),
                e.weight * global_decay * math.sin(e.angle_rad),
            )
            d_samp = max(0, int(round(e.delay_s * sr)))
            if d_samp > 0:
                if d_samp not in delay_weights:
                    delay_weights[d_samp] = torch.zeros(
                        self.N, self.N, dtype=_CDTYPE, device=device
                    )
                delay_weights[d_samp][di, si] += cw
                continue
            # Zero-delay edge — linear or saturating?
            sat_fn = _make_sat_fn(getattr(e, "saturation", ""),
                                  float(getattr(e, "saturation_knee", 1.0)))
            if sat_fn is not None:
                cw_t = torch.tensor(cw, dtype=_CDTYPE, device=device)
                self._sat_edges.append((di, si, cw_t, sat_fn))
            else:
                W_lin[di, si] += cw

        IW = torch.eye(self.N, dtype=_CDTYPE, device=device) - W_lin
        self.M: Tensor = _safe_inverse(IW)
        self._W_lin: Tensor = W_lin   # kept for the iterative correction pass

        # Delay ring buffers
        self._ring_W:   list[Tensor] = []
        self._ring_buf: list[Tensor] = []
        self._ring_d:   list[int]    = []
        self._ring_pos: list[int]    = []

        for d_samp, Wd in sorted(delay_weights.items()):
            self._ring_W.append(Wd)
            self._ring_buf.append(
                torch.zeros(batch_size, d_samp, self.N, dtype=_CDTYPE, device=device)
            )
            self._ring_d.append(d_samp)
            self._ring_pos.append(0)

    # ------------------------------------------------------------------

    def step(self, src: Tensor) -> Tensor:
        """Advance one sample.  src: (N,) or (B, N) → same shape."""
        single = src.dim() == 1
        x = src.unsqueeze(0) if single else src   # (B, N)

        # Accumulate delayed ring-buffer contributions
        if self._ring_W:
            delayed = torch.zeros_like(x)
            for Wd, buf, d, rpos in zip(
                self._ring_W, self._ring_buf, self._ring_d, self._ring_pos
            ):
                delayed = delayed + buf[:, rpos, :] @ Wd.T
            x = x + delayed

        # Linear solve — initial guess (exact if no saturating edges)
        X = x @ self.M.T
        _softclip_complex(X)

        # Iterative saturation correction
        if self._sat_edges:
            X = self._sat_iterate(x, X)

        # Commit to ring buffers
        for i, (buf, d) in enumerate(zip(self._ring_buf, self._ring_d)):
            rpos = self._ring_pos[i]
            buf[:, rpos, :] = X
            self._ring_pos[i] = (rpos + 1) % d

        return X.squeeze(0) if single else X

    def _sat_iterate(self, rhs: Tensor, X_init: Tensor) -> Tensor:
        """Fixed-point iteration for saturating edges.

        rhs     : (B, N) — sources + delayed contributions (fixed this sample)
        X_init  : (B, N) — linear pre-solve (starting point)

        Returns X : (B, N).  Sets last_saturated and last_convergence_iters.
        """
        X = X_init.clone()
        inf_t = self.infinity_threshold
        eps   = self.convergence_eps
        sat   = False

        for it in range(self.max_iterations):
            X_prev = X

            # Linear contribution from W_lin (fast matmul) + rhs
            X_new = rhs + X @ self._W_lin.T

            # Saturating contributions (per edge)
            for di, si, cw, sat_fn in self._sat_edges:
                X_new = X_new.clone()   # avoid in-place aliasing
                X_new[..., di] = X_new[..., di] + sat_fn(cw * X[..., si])

            # Infinity check — saturation fns bound the values,
            # but flag so the caller knows the cycle is at/near the limit
            if X_new.abs().max().item() > inf_t:
                sat = True
                X = X_new
                self.last_convergence_iters = it + 1
                break

            # Convergence
            diff = (X_new - X_prev).abs().max().item()
            X = X_new
            if diff < eps:
                self.last_convergence_iters = it + 1
                break
        else:
            self.last_convergence_iters = self.max_iterations

        self.last_saturated = sat
        return X

    # ------------------------------------------------------------------

    def output(self, X: Tensor, key: str) -> Tensor:
        """Extract signal for named node from a step() result.
        X: (N,) or (B, N) → scalar or (B,) complex128."""
        return X[..., self.ki[key]]

    def reset(self) -> None:
        """Zero all delay line state."""
        for buf in self._ring_buf:
            buf.zero_()
        for i in range(len(self._ring_pos)):
            self._ring_pos[i] = 0

    @classmethod
    def from_graph(
        cls,
        graph: "RoutingGraph",
        sr: float,
        device: torch.device,
        *,
        router_key: str = "",
        router_type: str = "",
        batch_size: int = 1,
        max_iterations: int = 64,
        convergence_eps: float = 1e-10,
        infinity_threshold: float = 1e6,
    ) -> "CompiledRouter":
        """Build a CompiledRouter from a RoutingGraph.

        router_key        — include only edges owned by this router instance.
        router_type       — filter node participation by source/sink rules.
        max_iterations    — cap on per-sample saturation iterations.
        convergence_eps   — |ΔX| threshold for early convergence exit.
        infinity_threshold — |X| threshold that flags saturation and cancels.
        """
        meta_edges = lower_meta_edges(graph)
        all_edges = list(graph.edges) + meta_edges

        if router_key:
            edges = [e for e in all_edges
                     if e.router_key == router_key or e.router_key == ""]
        else:
            edges = list(all_edges)

        if router_type:
            edges = [e for e in edges
                     if graph.is_source_in_router(e.src_key, router_type)
                     and graph.is_sink_in_router(e.dst_key, router_type)]

        referenced: set[str] = set()
        for e in edges:
            referenced.add(e.src_key)
            referenced.add(e.dst_key)
        node_keys = [k for k in graph.node_keys() if k in referenced]

        decay = (max(0.0, 1.0 - float(graph.feedback.decay))
                 if graph.feedback.enabled else 1.0)
        return cls(
            node_keys, edges, sr, device,
            global_decay=decay,
            batch_size=batch_size,
            max_iterations=max_iterations,
            convergence_eps=convergence_eps,
            infinity_threshold=infinity_threshold,
        )

    @classmethod
    def from_router_instance(
        cls,
        router: "RouterInstance",
        sr: float,
        device: torch.device,
        *,
        batch_size: int = 1,
        max_iterations: int = 64,
        convergence_eps: float = 1e-10,
        infinity_threshold: float = 1e6,
    ) -> "CompiledRouter":
        """Build a CompiledRouter directly from a RouterInstance.

        Uses the router's own graph, key ownership, and router_type
        participation rules so the compiled torch solve matches the editor's
        router-instance semantics.
        """
        return cls.from_graph(
            router.graph,
            sr,
            device,
            router_key=str(getattr(router, "key", "")),
            router_type=str(getattr(router, "router_type", "")),
            batch_size=batch_size,
            max_iterations=max_iterations,
            convergence_eps=convergence_eps,
            infinity_threshold=infinity_threshold,
        )
