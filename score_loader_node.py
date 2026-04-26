"""score_loader_node.py — ScoreLoaderNode: replays a saved .score.json arrangement.

Replaces TorchComposerNode when you want to render a pre-composed arrangement
without re-running the composition pipeline.  The node presents the same
score_out subscription contract so InstrumentNode / DriverNode consumers see
no difference.

Usage
-----
    from score_loader_node import ScoreLoaderNode

    loader = ScoreLoaderNode("composer", score_path="my_arrangement.score.json")

    compiled = compile_nodes(nodes=[loader.build_node(), instrument.build_node(), ...], ...)
    solver.reset()
    solver.dispatch_before_start()
    outputs = solver.run_schedule({}, n_frames=N_FRAMES)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from graph_solver import TensorNode, _CDTYPE
from score_persist import load_score


class ScoreLoaderNode(nn.Module):
    """Graph node that replays a pre-serialized atom list.

    Loads atoms once at construction time.  During dispatch_before_start the
    hook writes those atoms to the score_out FIFO for every consumer, exactly
    as TorchComposerNode would.

    Parameters
    ----------
    key:
        Node key (must match whatever InstrumentNode / DriverNode expects as
        the upstream composer key, typically "composer").
    score_path:
        Path to a .score.json written by score_persist.save_score().
    fifo_bank:
        Shared EdgeFifoBank.  Must be the same bank passed to compile_nodes /
        the consumer voice nodes.
    layer:
        Graph layer string.
    """

    def __init__(
        self,
        key: str,
        score_path: str | Path,
        *,
        fifo_bank: Any = None,
        layer: str = "score",
    ) -> None:
        super().__init__()
        self.key   = str(key)
        self.layer = str(layer)

        self._score_path = Path(score_path)
        self._atoms, self._meta = load_score(self._score_path)

        self.fifo_bank = fifo_bank
        self.subscription_ports: dict = {}

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        """Metadata from the score file (sample_rate, n_frames, tempo, etc.)."""
        return dict(self._meta)

    @property
    def atoms(self) -> list:
        """The loaded PerformanceAtom list (read-only view)."""
        return list(self._atoms)

    # ──────────────────────────────────────────────────────────────────────────

    def build_node(self) -> TensorNode:
        """Return a TensorNode that replays atoms during dispatch_before_start."""
        node = self

        def _hook() -> None:
            consumers: dict = (
                node.subscription_ports.get("score_out", {}).get("consumers", {})
            )
            if not consumers or not node._atoms:
                return
            if node.fifo_bank is None:
                return

            for voice_node_key, contract_list in consumers.items():
                dst = contract_list[0] if contract_list else {}
                slot_key = str(
                    dst.get("object_fifo_key")
                    or f"{node.key}_{voice_node_key}_atoms"
                )
                if not node.fifo_bank.has_slot(slot_key):
                    node.fifo_bank.claim_object(slot_key, fifo_size=16)
                node.fifo_bank.write_batch(slot_key, list(node._atoms))

        def _transform(x: Tensor) -> Tensor:
            return x.to(_CDTYPE)

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=_transform,
            fire_before_start=True,
            hook_before_start=_hook,
            analytic_module=self,
            subscription_ports=("score_out",),
            subscription_contracts={
                "score_out": {
                    "kind":               "score_producer",
                    "transport":          "object_fifo",
                    "object_fifo_suffix": "atoms",
                },
            },
        )
