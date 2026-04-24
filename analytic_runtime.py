"""Runtime render boundary for analytic patch engines.

This module is intentionally engine-neutral.  A render produces named products;
stereo buses are compatibility views over those products, not the contract
itself.  That keeps the graph path free to expose arbitrary TensorNode products
without forcing everything through left/right audio semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

_CDTYPE = torch.complex128


@dataclass(frozen=True)
class RenderProduct:
    """One named runtime product emitted by an engine."""

    key: str
    tensor: torch.Tensor
    semantic: str = ""
    origin_node: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def numpy(self) -> np.ndarray:
        arr = self.tensor.detach().cpu().numpy()
        return np.asarray(arr)


@dataclass
class PatchRenderResult:
    """Engine-neutral render result.

    ``products`` is the primary surface.  Compatibility fields mirror the old
    legacy render API so call sites can migrate without losing current behavior.
    """

    engine: str
    sample_rate: int
    n_samples: int
    products: dict[str, RenderProduct] = field(default_factory=dict)
    output_product_keys: list[str] = field(default_factory=list)
    sidecar: Any = None
    mixer_products: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    compat_left: Optional[np.ndarray] = None
    compat_right: Optional[np.ndarray] = None
    compat_output_bus: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def product_tensor(self, key: str) -> torch.Tensor:
        return self.products[key].tensor

    def product_numpy(self, key: str) -> np.ndarray:
        return self.products[key].numpy()

    def output_bus(self) -> np.ndarray:
        if self.compat_output_bus is not None:
            return np.asarray(self.compat_output_bus)
        if not self.output_product_keys:
            return np.zeros((self.n_samples, 0), dtype=np.float32)
        cols = []
        for key in self.output_product_keys:
            if key not in self.products:
                continue
            arr = self.product_numpy(key)
            cols.append(np.asarray(arr).reshape(-1)[: self.n_samples].real.astype(np.float32, copy=False))
        if not cols:
            return np.zeros((self.n_samples, 0), dtype=np.float32)
        return np.column_stack(cols).astype(np.float32, copy=False)

    def stereo_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if self.compat_left is not None and self.compat_right is not None:
            return np.asarray(self.compat_left), np.asarray(self.compat_right)
        bus = self.output_bus()
        if bus.shape[1] == 0:
            z = np.zeros(self.n_samples, dtype=np.float32)
            return z, z
        left = bus[:, 0]
        right = bus[:, 1] if bus.shape[1] > 1 else bus[:, 0]
        return left, right

    def legacy_tuple(
        self,
        *,
        return_output_channels: bool = False,
        return_mixer_sigs: bool = False,
        return_sidecar: bool = False,
    ) -> tuple:
        left, right = self.stereo_pair()
        ret: list[Any] = [left, right]
        if return_output_channels:
            ret.append(self.output_bus())
        if return_mixer_sigs:
            ret.append(self.mixer_products)
        if return_sidecar:
            ret.append(self.sidecar)
        return tuple(ret)


def _as_tensor_product(
    key: str,
    value: np.ndarray | torch.Tensor,
    *,
    semantic: str,
    origin_node: str = "",
    dtype: Optional[torch.dtype] = None,
) -> RenderProduct:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
        if dtype is not None:
            tensor = tensor.to(dtype)
    else:
        arr = np.asarray(value)
        if dtype is None:
            dtype = _CDTYPE if np.iscomplexobj(arr) else torch.float32
        tensor = torch.as_tensor(arr, dtype=dtype)
    return RenderProduct(
        key=key,
        tensor=tensor,
        semantic=semantic,
        origin_node=origin_node or key,
    )


def render_patch_legacy(
    patch,
    *,
    granular_seed_offset: int = 0,
    file_render: bool = False,
) -> PatchRenderResult:
    """Render with the legacy engine and expose it through the product contract."""
    from analytic_synth_legacy import _synthesize_patch

    left, right, output_bus, mixer_outs, sidecar = _synthesize_patch(
        patch,
        granular_seed_offset=granular_seed_offset,
        file_render=file_render,
        _return_output_channels=True,
        _return_mixer_sigs=True,
        _return_sidecar=True,
    )
    sr = int(getattr(patch, "preview_sr", 48_000))
    n = int(np.asarray(output_bus).shape[0])
    products: dict[str, RenderProduct] = {
        "__legacy_left__": _as_tensor_product("__legacy_left__", left, semantic="compat_audio_channel"),
        "__legacy_right__": _as_tensor_product("__legacy_right__", right, semantic="compat_audio_channel"),
        "__legacy_output_bus__": _as_tensor_product("__legacy_output_bus__", output_bus, semantic="compat_output_bus"),
    }
    for key, (ml, mr) in dict(mixer_outs or {}).items():
        products[f"{key}:left"] = _as_tensor_product(f"{key}:left", ml, semantic="compat_mixer_channel", origin_node=key)
        products[f"{key}:right"] = _as_tensor_product(f"{key}:right", mr, semantic="compat_mixer_channel", origin_node=key)
    return PatchRenderResult(
        engine="legacy",
        sample_rate=sr,
        n_samples=n,
        products=products,
        output_product_keys=["__legacy_output_bus__"],
        sidecar=sidecar,
        mixer_products=dict(mixer_outs or {}),
        compat_left=np.asarray(left, dtype=np.float32),
        compat_right=np.asarray(right, dtype=np.float32),
        compat_output_bus=np.asarray(output_bus, dtype=np.float32),
    )


def _resolve_product_keys(compiled, patch, requested: Optional[Iterable[str]]) -> list[str]:
    if requested is not None:
        return [str(k) for k in requested]
    keys: list[str] = []
    keys.extend(str(k) for k in getattr(compiled, "output_node_keys", []))
    for mixer in getattr(patch, "mixers", []):
        key = str(getattr(mixer, "key", ""))
        if key and bool(getattr(mixer, "projection_active", False)):
            keys.append(key)
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def render_patch_graph(
    patch,
    *,
    sample_rate: Optional[float] = None,
    n_samples: Optional[int] = None,
    product_keys: Optional[Iterable[str]] = None,
    use_cache: bool = False,
    profile: bool = False,
) -> PatchRenderResult:
    """Render named graph products with ``GraphSolver.step``.

    The returned products are raw node products.  No assumption is made that
    they are stereo, PCM, or even final outputs.
    """
    from network_materializer import compile_network
    from graph_solver import _T

    t_total0 = time.perf_counter()
    timings: dict[str, float | int | bool] = {}
    sr = int(sample_rate if sample_rate is not None else getattr(patch, "preview_sr", 48_000))
    n = int(n_samples if n_samples is not None else int(sr * float(getattr(patch, "duration", 0.0))))
    if n <= 0:
        n = 1
    if profile:
        _T.reset()

    compiled = None
    cache_hit = False
    t_compile0 = time.perf_counter()
    if use_cache:
        sig = (sr, tuple(product_keys or ()))
        if getattr(patch, "_compiled_signature", None) == sig:
            compiled = getattr(patch, "_compiled", None)
            cache_hit = compiled is not None
        if compiled is None:
            compiled = compile_network(patch, sr)
            patch._compiled = compiled
            patch._compiled_signature = sig
    else:
        compiled = compile_network(patch, sr)
    timings["compile_s"] = time.perf_counter() - t_compile0
    timings["cache_hit"] = cache_hit

    solver = compiled.solver
    t_reset0 = time.perf_counter()
    solver.reset()
    timings["reset_s"] = time.perf_counter() - t_reset0
    t_keys0 = time.perf_counter()
    keys = _resolve_product_keys(compiled, patch, product_keys)
    buckets: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    timings["resolve_products_s"] = time.perf_counter() - t_keys0

    t_step0 = time.perf_counter()
    with torch.no_grad():
        for _idx in range(n):
            outputs = solver.step({})
            for key in keys:
                value = outputs.get(key)
                if value is None:
                    value = torch.zeros((), dtype=_CDTYPE)
                buckets[key].append(value.detach().to(_CDTYPE).reshape(()))
    timings["step_s"] = time.perf_counter() - t_step0

    t_stack0 = time.perf_counter()
    products: dict[str, RenderProduct] = {}
    for key, values in buckets.items():
        tensor = torch.stack(values).to(_CDTYPE)
        products[key] = RenderProduct(
            key=key,
            tensor=tensor,
            semantic="graph_product",
            origin_node=key,
        )
    timings["stack_products_s"] = time.perf_counter() - t_stack0
    timings["total_s"] = time.perf_counter() - t_total0
    timings["avg_step_ms"] = (float(timings["step_s"]) / max(n, 1)) * 1000.0
    timings["samples"] = n
    timings["product_count"] = len(keys)
    timings["node_count"] = len(getattr(solver, "nodes", ()))
    timings["edge_count"] = len(getattr(solver, "edges", ()))
    if profile:
        _T.report(title="Graph shadow solver profile")
    return PatchRenderResult(
        engine="graph",
        sample_rate=sr,
        n_samples=n,
        products=products,
        output_product_keys=keys,
        metadata={
            "module_registry_keys": sorted(getattr(compiled, "module_registry", {}).keys()),
            "timings": timings,
        },
    )


def graph_shadow_summary(
    legacy: PatchRenderResult,
    graph: PatchRenderResult,
) -> dict[str, float | int | str]:
    """Return a compact parity summary without deciding pass/fail policy."""
    legacy_bus = legacy.output_bus()
    graph_bus = graph.output_bus()
    n = min(len(legacy_bus), len(graph_bus))
    ch = min(legacy_bus.shape[1] if legacy_bus.ndim == 2 else 0,
             graph_bus.shape[1] if graph_bus.ndim == 2 else 0)
    if n == 0 or ch == 0:
        return {"samples": int(n), "channels": int(ch), "rms_delta": float("nan"), "peak_delta": float("nan")}
    delta = legacy_bus[:n, :ch].astype(np.float64) - graph_bus[:n, :ch].astype(np.float64)
    return {
        "samples": int(n),
        "channels": int(ch),
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "peak_delta": float(np.max(np.abs(delta))),
    }


def clear_compiled_graph_cache(patch) -> None:
    patch._compiled = None
    patch._compiled_signature = ()
