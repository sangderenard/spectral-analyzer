"""
c_transforms — Python side of the C transform registry.

CTransformID mirrors the C enum in csrc/include/transforms_api.h.
Keep the two in sync when adding new transform types.

Dispatch contract
-----------------
A CyclicTensorBlock is C-dispatchable (for the Picard step) when:

  * Every internal edge has saturation_fn=None (no arbitrary Python callable)
    and saturation_policy in the C registry.
  * Every node has transform=None (identity accumulator) OR its callable
    carries a ``c_transform_key`` attribute whose value is in the registry.

When the block is dispatchable, try_compile_picard() returns a compiled
PicardSCC handle (from _spectral_kernels).  The block stores this handle
and calls it instead of the Python _apply_once() loop.
"""
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from graph_solver import CyclicTensorBlock


class CTransformID(IntEnum):
    IDENTITY = 0
    TANH     = 1
    SOFTCLIP = 2
    HARDCLIP = 3


# Maps TensorEdge.saturation_policy strings → C enum values.
_EDGE_SAT_REGISTRY: dict[str, int] = {
    "":         CTransformID.IDENTITY,
    "tanh":     CTransformID.TANH,
    "softclip": CTransformID.SOFTCLIP,
    "hardclip": CTransformID.HARDCLIP,
}

# Maps c_transform_key attribute values → C enum values.
_NODE_TF_REGISTRY: dict[str, int] = {
    "identity":  CTransformID.IDENTITY,
    "tanh":      CTransformID.TANH,
    "softclip":  CTransformID.SOFTCLIP,
    "hardclip":  CTransformID.HARDCLIP,
}


def _edge_sat_id(edge) -> Optional[int]:
    """CTransformID for edge saturation, or None if not C-dispatchable."""
    if edge.saturation_fn is not None:
        return None
    return _EDGE_SAT_REGISTRY.get(edge.saturation_policy or "")


def _node_tf_id(transform: Optional[Callable]) -> Optional[int]:
    """CTransformID for a node transform, or None if not C-dispatchable."""
    if transform is None:
        return CTransformID.IDENTITY
    key = getattr(transform, "c_transform_key", None)
    return _NODE_TF_REGISTRY.get(key)


def compile_singleton_picard(transform: Optional[Callable], tol: float = 1e-10):
    """
    Compile a PicardSCC(N=1, no edges, max_iterations=1) for a singleton
    feedforward node whose transform is in the C registry.

    Returns the compiled handle or None if the transform is not C-registered
    or _spectral_kernels is unavailable.  With no internal edges the kernel
    runs exactly one pass (feedforward, no convergence iteration needed).
    """
    try:
        from _spectral_kernels import PicardSCC
    except ImportError:
        return None

    import numpy as np
    tf_id = _node_tf_id(transform)
    if tf_id is None:
        return None

    knee = float(getattr(transform, "c_transform_knee", 1.0)) if transform is not None else 1.0
    empty_i = np.zeros(0, dtype=np.int32)
    empty_f = np.zeros(0, dtype=np.float64)

    return PicardSCC(
        N             = 1,
        src_idxs      = empty_i,
        dst_idxs      = empty_i,
        edge_w_re     = empty_f,
        edge_w_im     = empty_f,
        edge_sat_ids  = empty_i,
        edge_knees    = empty_f,
        node_tf_ids   = np.array([int(tf_id)], dtype=np.int32),
        node_knees    = np.array([knee],        dtype=np.float64),
        max_iterations  = 1,
        convergence_tol = tol,
    )


def try_compile_picard(block: "CyclicTensorBlock"):
    """
    Attempt to compile a PicardSCC handle for the block.

    Returns the compiled handle on success, or None if any transform is
    not in the C registry or the _spectral_kernels extension is unavailable.
    """
    try:
        from _spectral_kernels import PicardSCC
    except ImportError:
        return None

    import torch
    _CDTYPE = torch.complex128

    N     = len(block.node_keys)
    index = {key: i for i, key in enumerate(block.node_keys)}

    # ── Compile node transforms ───────────────────────────────────────────
    node_tf_ids:  list[int]   = []
    node_knees:   list[float] = []
    for key in block.node_keys:
        tf = block.node_map[key].transform
        tf_id = _node_tf_id(tf)
        if tf_id is None:
            return None
        node_tf_ids.append(int(tf_id))
        node_knees.append(float(getattr(tf, "c_transform_knee", 1.0)) if tf is not None else 1.0)

    # ── Compile edges ─────────────────────────────────────────────────────
    src_idxs:    list[int]   = []
    dst_idxs:    list[int]   = []
    edge_w_re:   list[float] = []
    edge_w_im:   list[float] = []
    edge_sat_ids: list[int]  = []
    edge_knees:  list[float] = []

    def _to_cd(v) -> complex:
        t = torch.as_tensor(v, dtype=_CDTYPE)
        return complex(t.item())

    for edge in block.internal_edges:
        sat_id = _edge_sat_id(edge)
        if sat_id is None:
            return None

        # Fold weight * analog_complex_delay * masks into one complex scalar.
        # Crosstalk adds to the combined weight (both act linearly on z[src]).
        w = (_to_cd(edge.weight)
             * _to_cd(edge.analog_complex_delay)
             * _to_cd(edge.src_mask)
             * _to_cd(edge.dst_mask)
             * _to_cd(edge.activity_mask)
             + _to_cd(edge.crosstalk_weight))

        src_idxs.append(index[edge.src_key])
        dst_idxs.append(index[edge.dst_key])
        edge_w_re.append(w.real)
        edge_w_im.append(w.imag)
        edge_sat_ids.append(int(sat_id))
        edge_knees.append(float(edge.saturation_knee))

    n_edges = len(src_idxs)

    def _int32(lst: list[int]) -> np.ndarray:
        return np.array(lst, dtype=np.int32)

    def _f64(lst: list[float]) -> np.ndarray:
        return np.array(lst, dtype=np.float64)

    return PicardSCC(
        N            = N,
        src_idxs     = _int32(src_idxs),
        dst_idxs     = _int32(dst_idxs),
        edge_w_re    = _f64(edge_w_re),
        edge_w_im    = _f64(edge_w_im),
        edge_sat_ids = _int32(edge_sat_ids),
        edge_knees   = _f64(edge_knees),
        node_tf_ids  = _int32(node_tf_ids),
        node_knees   = _f64(node_knees),
        max_iterations  = block.K_max,
        convergence_tol = block.tol,
    )
