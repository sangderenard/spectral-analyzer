"""kernel_fn.py — Python container for serial C daemon kernels.

Architecture
------------
CKernelFunction is the base class for all C-backed serial kernel operations.
It extends torch.autograd.Function so every forward pass automatically
participates in PyTorch's autograd graph — but the backward MUST be written
by hand, because no gradient tape crosses the C boundary.

The pattern:

    class MyKernel(CKernelFunction):
        @staticmethod
        def forward(ctx, state: CKernelState, inputs: torch.Tensor) -> torch.Tensor:
            # 1. Copy tensor data to state (or use pre-loaded params)
            # 2. Call state.run() or state.step()
            # 3. Save whatever backward needs via ctx.save_for_backward
            ...

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            # Write the transpose/adjoint of the forward operation.
            # ctx returns None for non-differentiable arguments.
            ...

CKernelState
------------
Wraps the pybind11 _spectral_kernels.RouterStep (or any other C daemon) and
exposes torch Tensors as the surface.  Internally it maintains a numpy view of
the tensors' data so the C code can operate directly without an extra copy.

RouterStepKernel
----------------
The concrete CKernelFunction for CompiledRouter.step().

Forward:  X = (src + delayed_contrib) @ M.T
Backward: grad_src = grad_X @ M   (M = M.T.T, i.e. un-transposing the solve)

The ring-buffer state is treated as non-differentiable (it is opaque stateful
memory).  Only the linear path through M is differentiated.  This matches the
physical interpretation: delay-line state is a temporal side effect, not a
learnable parameter at inference time.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

# ── Extension loading ──────────────────────────────────────────────────────
# Try the pybind11 module first; fall back to ctypes.  The pybind11 build
# produces _spectral_kernels.pyd (Windows) or _spectral_kernels.so (Linux/Mac)
# in the package root.

_EXT_DIR = os.path.dirname(__file__)

try:
    import _spectral_kernels as _C  # type: ignore[import]
    _BACKEND = "pybind11"
except ImportError:
    _C = None
    _BACKEND = "unavailable"
    # The ctypes fallback is handled by CKernelState if needed


def backend() -> str:
    """Return the active backend: 'pybind11' or 'unavailable'."""
    return _BACKEND


# ── Helper: complex128 tensor ↔ float64 numpy interleaved view ────────────

def _as_interleaved(t: Tensor) -> np.ndarray:
    """Return a float64 numpy view of a complex128 tensor (zero-copy).

    torch.complex128 stores [re, im] pairs contiguously, so .view(torch.float64)
    gives exactly the interleaved layout the C kernels expect.
    """
    if not t.is_contiguous():
        t = t.contiguous()
    return t.view(torch.float64).numpy()


def _from_interleaved(arr: np.ndarray, shape: tuple) -> Tensor:
    """Wrap an interleaved float64 numpy array as a complex128 tensor."""
    t = torch.from_numpy(arr.reshape(-1))
    return t.view(torch.complex128).reshape(shape)


# ── CKernelState: owns the C daemon handle ────────────────────────────────

class CKernelState:
    """Python owner of a C daemon handle.

    Hold one per router instance (one per CompiledRouter that you want to run
    through the C path).  The state is mutable — ring buffers live inside the C
    heap.

    Parameters
    ----------
    compiled_router : CompiledRouter
        A fully initialised CompiledRouter whose matrices will be copied into
        the C daemon.  The daemon does NOT share memory with the router; changes
        to the router after this point are not reflected.
    batch_size : int
        Number of independent instances.  Can be changed later via resize_batch.
    """

    def __init__(self, compiled_router: Any, batch_size: int = 1) -> None:
        if _C is None:
            raise RuntimeError(
                "C kernel extension not available. "
                "Build with: cmake csrc/ && cmake --build build/ "
                "or check that _spectral_kernels is on sys.path."
            )
        router = compiled_router
        N = router.N
        B = batch_size

        M   = router.M.to(torch.complex128).contiguous()
        W   = router._W_lin.to(torch.complex128).contiguous()

        M_re = M.real.numpy().astype(np.float64, copy=False)
        M_im = M.imag.numpy().astype(np.float64, copy=False)
        W_re = W.real.numpy().astype(np.float64, copy=False)
        W_im = W.imag.numpy().astype(np.float64, copy=False)

        # Delay groups from the CompiledRouter ring state
        delay_lengths: list[int] = list(router._ring_d)
        n_delays = len(delay_lengths)

        if n_delays > 0:
            ring_W_stack = torch.stack([rw.to(torch.complex128) for rw in router._ring_W])
            rW_re = ring_W_stack.real.numpy().astype(np.float64, copy=False)
            rW_im = ring_W_stack.imag.numpy().astype(np.float64, copy=False)
        else:
            rW_re = np.zeros((0, N, N), dtype=np.float64)
            rW_im = np.zeros((0, N, N), dtype=np.float64)

        self._handle = _C.RouterStep(
            N, B,
            M_re.ravel(), M_im.ravel(),
            W_re.ravel(), W_im.ravel(),
            delay_lengths,
            rW_re.ravel(), rW_im.ravel(),
        )
        self.N = N
        self.B = B
        # Keep a reference to M for the backward pass
        self.M: Tensor = M

    # ── Stateful step methods (bypass autograd — use RouterStepKernel for grad)

    def step_numpy(self, src: np.ndarray) -> np.ndarray:
        """Advance one sample.  src: float64 (B, N, 2) interleaved → same."""
        return self._handle.step(src.ravel()).reshape(self.B, self.N, 2)

    def run_numpy(self, src: np.ndarray) -> np.ndarray:
        """Advance T samples.  src: float64 (T, B, N, 2) → same."""
        T = src.shape[0]
        return self._handle.run(src.ravel()).reshape(T, self.B, self.N, 2)

    def reset(self) -> None:
        """Zero all ring-buffer delay state."""
        self._handle.reset()

    def resize_batch(self, new_B: int) -> None:
        self._handle.resize_batch(new_B)
        self.B = new_B

    def diagnostics(self) -> dict:
        return self._handle.diagnostics()


# ── CKernelFunction base ───────────────────────────────────────────────────

class CKernelFunction(torch.autograd.Function):
    """Base class for all C-backed serial kernel autograd functions.

    Subclasses MUST override both forward() and backward().  The base class
    does not provide a default implementation for either — this is intentional:
    every kernel has a different forward operation and therefore a different
    adjoint.  There is no automatic differentiation through the C boundary.

    Convention
    ----------
    - forward() receives a CKernelState as the first non-ctx argument.  The
      state is NOT saved via ctx.save_for_backward (it's not a Tensor).  Store
      whatever you need for the backward on ctx directly.
    - backward() returns one None per non-Tensor argument (i.e. the first
      return value, for the state, is always None).

    Example skeleton::

        class MyKernel(CKernelFunction):
            @staticmethod
            def forward(ctx, state, x):
                ctx.state = state
                ctx.save_for_backward(x)
                return state.run_numpy(x.numpy())   # or step_numpy

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                grad_x = ...   # write the adjoint of your forward here
                return None, grad_x   # None = no gradient for `state`
    """

    @staticmethod
    def forward(ctx: Any, state: CKernelState, *args: Any) -> Any:
        raise NotImplementedError(
            "CKernelFunction subclasses must implement forward()"
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError(
            "CKernelFunction subclasses must implement backward() — "
            "no gradient tape crosses the C boundary."
        )


# ── RouterStepKernel: differentiable wrapper ──────────────────────────────

class RouterStepKernel(CKernelFunction):
    """Differentiable single-sample router step through the C daemon.

    Forward
    -------
    Delegating exactly to CompiledRouter.step() but through the C fast path:

        rhs = src + sum_d( ring_state_d )      # accumulated in C
        X   = rhs @ M.T                        # linear solve

    Backward
    --------
    The only differentiable path is the linear matmul X = rhs @ M.T.
    Treating ring-buffer contributions as non-differentiable (they are opaque
    causal history; not meaningful to backprop through time here):

        dL/d_rhs = dL/dX @ M        # (M.T).T = M
        dL/d_src = dL/d_rhs         # src feeds rhs directly

    Ring-buffer state gradient: None (treated as frozen history).
    M gradient: None (M is a compiled parameter, not being learned here).

    Usage
    -----
        out = RouterStepKernel.apply(state, src)
        out.backward(grad)     # grad flows back to src

    Note: call state.reset() between independent sequences; the ring-buffer
    state mutation inside the C daemon is not reversed by backward().
    """

    @staticmethod
    def forward(ctx: Any, state: CKernelState, src: Tensor) -> Tensor:
        """Advance one sample.

        Parameters
        ----------
        state : CKernelState
        src   : complex128 Tensor of shape (N,) or (B, N)

        Returns
        -------
        X : complex128 Tensor, same shape as src.
        """
        single = src.dim() == 1
        if single:
            if state.B != 1:
                raise ValueError(
                    f"router_step: single-vector src requires state.B==1, got B={state.B}. "
                    "Pass a (B, N) tensor or create CKernelState with batch_size=1."
                )
            src = src.unsqueeze(0)   # (1, N)

        actual_B = src.shape[0]
        if actual_B != state.B:
            raise ValueError(
                f"router_step: src batch dim {actual_B} != state.B {state.B}"
            )

        src_c = src.to(torch.complex128).contiguous()
        arr   = _as_interleaved(src_c).reshape(actual_B, state.N, 2)
        out_arr = state.step_numpy(arr)
        X = _from_interleaved(out_arr, (actual_B, state.N))

        # Save for backward — only what the gradient computation needs.
        # The M matrix lives on ctx.state; store it directly.
        ctx.save_for_backward(state.M)
        ctx.single = single

        return X.squeeze(0) if single else X

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple:
        """Backprop through the linear solve X = rhs @ M.T.

        grad_rhs = grad_X @ M   (transpose of M.T is M)
        grad_src = grad_rhs     (src passes through to rhs additively)

        Returns (None, grad_src) — None for the state argument.
        """
        (M,) = ctx.saved_tensors
        single = ctx.single

        go = grad_output.to(torch.complex128)
        if single:
            go = go.unsqueeze(0)

        # X = rhs @ M.T  →  dL/drhs = dL/dX @ M
        grad_src = go @ M

        if single:
            grad_src = grad_src.squeeze(0)

        return None, grad_src   # (state gradient, src gradient)


# ── Convenience: apply wrappers ───────────────────────────────────────────

def router_step(state: CKernelState, src: Tensor) -> Tensor:
    """Run one router sample through the C daemon with autograd support.

    Equivalent to CompiledRouter.step() but:
    - inner loop executes in C (no Python per-step overhead)
    - gradients flow back through src via the hand-written backward

    Parameters
    ----------
    state : CKernelState  — compiled router daemon state
    src   : Tensor        — complex128, shape (N,) or (B, N)

    Returns
    -------
    X : Tensor, same shape as src, complex128.
    """
    return RouterStepKernel.apply(state, src)


def router_run(state: CKernelState, src: Tensor) -> Tensor:
    """Run T samples serially through the C daemon.

    This does NOT flow gradients sample-by-sample; if you need BPTT use
    router_step() in a Python loop.  This function is for inference speed
    where only the final output gradient matters.

    Parameters
    ----------
    state : CKernelState
    src   : Tensor — complex128, shape (T, N) or (T, B, N)

    Returns
    -------
    out : Tensor — same shape as src.
    """
    batched = src.dim() == 3
    if src.dim() == 2:
        src = src.unsqueeze(1)   # (T, 1, N)
    T, B, N = src.shape
    src_c = src.to(torch.complex128).contiguous()
    arr   = _as_interleaved(src_c).reshape(T, B, N, 2)
    out_arr = state.run_numpy(arr)
    out = _from_interleaved(out_arr, (T, B, N))
    if not batched:
        out = out.squeeze(1)
    return out
