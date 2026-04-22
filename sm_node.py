"""sm_node.py — StateMachineNode abstract base class.

State machine nodes are parallel identity-preserving signal processors.
They are NOT aggregating mixers and do NOT appear in the routing weight
matrix W.  Instead, their named I/O slots are registered as ordinary
graph nodes with explicit participation rules, and the SM advances its
internal state between layer solves.

Signal identity contract
------------------------
Every input slot receives exactly one signal keyed by name (via item_slot
on the feeding edge).  Signals are never summed before reaching the SM.
The SM may cross-couple slots internally (e.g. sympathetic string resonance,
multi-driver instrument body), but each slot's identity is always preserved
at the boundary.

Torch contract
--------------
All signals are torch.complex128.  Scalars are 0-d tensors; batched signals
are (B,) tensors.  The SM is responsible for handling both shapes.

Graph registration
------------------
``register_in_graph`` inserts up to three sets of graph nodes for each SM
instance:

  {sm_key}.in.{slot}   — sink node in the layer feeding this SM.
                          Participation: sink_router_types = [feeding_layer_router]
                          Holds the signal from the previous layer solve.

  {sm_key}.out.{slot}  — source node in the layer fed by this SM.
                          Participation: source_router_types = [receiving_layer_router]
                          Holds the SM output until the next layer solve reads it.

  {sm_key}.ctrl.{slot} — control source node in the parameter layer.
                          These nodes drive ParamEdges rather than signal edges.
                          They let an SM push intrinsic control state such as
                          aperture pressure or clan sympathy back into upstream
                          driver parameters without exposing SM internals to the
                          matrix solve.

The SM's step() reads from .in nodes and writes to .out nodes.  Optional
control outputs are written to .ctrl nodes.
The routing solver never touches SM-internal state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from routing_engine import ParamEdge, RoutingGraph

_CDTYPE = torch.complex128


class StateMachineNode(ABC):
    """Abstract base for state machine nodes in the signal graph.

    Subclass and implement ``input_slots``, ``output_slots``,
    ``sm_layer``, ``step``, and ``reset``.  Override
    ``control_output_slots`` and ``default_param_connections`` when the
    SM also emits control signals into the parameter-routing layer.

    Example subclass skeleton::

        class MyBodySM(StateMachineNode):
            sm_layer = "performer"

            @property
            def input_slots(self): return ["driver_0", "driver_1"]

            @property
            def output_slots(self): return ["aperture", "body_pressure"]

            def step(self, inputs):
                # inputs: {"driver_0": Tensor, "driver_1": Tensor}
                out = self._advance(inputs)
                return {"aperture": out[0], "body_pressure": out[1]}

            def reset(self): self._state = ...
    """

    # Override in subclass: which SM_LAYER this node belongs to.
    sm_layer: str = ""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def input_slots(self) -> list[str]:
        """Named input channels this SM accepts (order matters for batching)."""

    @property
    @abstractmethod
    def output_slots(self) -> list[str]:
        """Named output channels this SM emits."""

    @property
    def control_output_slots(self) -> list[str]:
        """Named control outputs this SM emits into the param layer."""
        return []

    @abstractmethod
    def step(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Advance one sample.

        Parameters
        ----------
        inputs : {slot_name: complex128 tensor}
            One entry per input_slot.  Missing slots arrive as zero.

        Returns
        -------
        {slot_name: complex128 tensor}
            One entry per output_slot.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state for reuse between renders."""

    # ------------------------------------------------------------------
    # Graph registration helpers
    # ------------------------------------------------------------------

    def in_node_key(self, sm_key: str, slot: str) -> str:
        return f"{sm_key}.in.{slot}"

    def out_node_key(self, sm_key: str, slot: str) -> str:
        return f"{sm_key}.out.{slot}"

    def ctrl_node_key(self, sm_key: str, slot: str) -> str:
        return f"{sm_key}.ctrl.{slot}"

    def default_param_connections(
        self,
        sm_key: str,
        *,
        target_map: "dict[str, list[tuple[str, str, float, int]]] | None" = None,
    ) -> list["ParamEdge"]:
        """Return default ParamEdges emitted by this SM.

        The base implementation accepts a simple *target_map*:

            {
                "aperture_pressure": [
                    ("driver_a", "chirp.f_delta_start", 1.0, 0),
                    ("driver_b", "chirp.f_delta_start", 0.8, 0),
                ],
            }

        Keys are control-output slot names.  Values are lists of
        ``(dst_key, param_path, weight, delay_samples)`` tuples.
        Subclasses may override this method for richer topology-aware wiring.
        """
        if not target_map:
            return []

        from routing_engine import ParamEdge

        param_edges: list[ParamEdge] = []
        for slot, targets in target_map.items():
            if slot not in self.control_output_slots:
                continue
            src_key = self.ctrl_node_key(sm_key, slot)
            for dst_key, param_path, weight, delay_samples in targets:
                param_edges.append(
                    ParamEdge(
                        src_key=src_key,
                        dst_key=str(dst_key),
                        weight=float(weight),
                        extractor="real",
                        delay_samples=max(0, int(delay_samples)),
                        param_path=str(param_path),
                    )
                )
        return param_edges

    def register_in_graph(
        self,
        graph: "RoutingGraph",
        sm_key: str,
        *,
        feeding_router_type: str = "",
        receiving_router_type: str = "",
        parameter_router_type: str = "",
        install_default_param_edges: bool = False,
        default_param_target_map: "dict[str, list[tuple[str, str, float, int]]] | None" = None,
    ) -> None:
        """Register this SM's I/O nodes in *graph*.

        Parameters
        ----------
        sm_key
            Unique key for this SM instance (e.g. "performer_0").
        feeding_router_type
            The router type whose edges write into this SM's input slots.
            Input nodes will be sinks only in that router.
        receiving_router_type
            The router type that reads this SM's output slots as sources.
            Output nodes will be sources only in that router.
        parameter_router_type
            Router type that reads this SM's control outputs as sources for
            parameter routing.  Control nodes remain sources-only.
        install_default_param_edges
            When True, append ``default_param_connections(...)`` into
            ``graph.param_edges``.
        default_param_target_map
            Optional wiring spec consumed by the base implementation of
            ``default_param_connections``.
        """
        for slot in self.input_slots:
            key = self.in_node_key(sm_key, slot)
            graph.add_node(
                key,
                node_type=self.sm_layer,
                source_router_types=[],
                sink_router_types=(
                    [feeding_router_type] if feeding_router_type else None
                ),
            )

        for slot in self.output_slots:
            key = self.out_node_key(sm_key, slot)
            graph.add_node(
                key,
                node_type=self.sm_layer,
                source_router_types=(
                    [receiving_router_type] if receiving_router_type else None
                ),
                sink_router_types=[],
            )

        for slot in self.control_output_slots:
            key = self.ctrl_node_key(sm_key, slot)
            graph.add_node(
                key,
                node_type=self.sm_layer,
                source_router_types=(
                    [parameter_router_type] if parameter_router_type else None
                ),
                sink_router_types=[],
            )

        if install_default_param_edges:
            graph.param_edges.extend(
                self.default_param_connections(
                    sm_key,
                    target_map=default_param_target_map,
                )
            )

    def deregister_from_graph(self, graph: "RoutingGraph", sm_key: str) -> None:
        """Remove this SM's I/O nodes from *graph*.

        Prunes any edges that reference these nodes.
        """
        keys = (
            [self.in_node_key(sm_key, s) for s in self.input_slots]
            + [self.out_node_key(sm_key, s) for s in self.output_slots]
            + [self.ctrl_node_key(sm_key, s) for s in self.control_output_slots]
        )
        key_set = set(keys)
        graph.nodes = [n for n in graph.nodes if n not in key_set]
        for d in (graph.node_types, graph.node_layers,
                  graph.node_source_router_types, graph.node_sink_router_types):
            for k in keys:
                d.pop(k, None)
        graph.prune_keys(set(graph.nodes))

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def read_inputs(
        self,
        graph_state: "dict[str, Tensor]",
        sm_key: str,
    ) -> dict[str, Tensor]:
        """Read this SM's input slot values from a graph state dict.

        *graph_state* maps node_key → Tensor, as produced by CompiledRouter
        after a layer solve.  Missing nodes return zero (complex128 scalar).
        """
        zero = torch.zeros((), dtype=_CDTYPE)
        return {
            slot: graph_state.get(self.in_node_key(sm_key, slot), zero)
            for slot in self.input_slots
        }

    def write_outputs(
        self,
        outputs: dict[str, Tensor],
        graph_state: "dict[str, Tensor]",
        sm_key: str,
    ) -> None:
        """Write this SM's outputs into a graph state dict in-place."""
        zero = torch.zeros((), dtype=_CDTYPE)
        for slot in self.output_slots:
            key = self.out_node_key(sm_key, slot)
            graph_state[key] = outputs.get(slot, zero)

    def write_control_outputs(
        self,
        outputs: dict[str, Tensor],
        graph_state: "dict[str, Tensor]",
        sm_key: str,
    ) -> None:
        """Write this SM's control outputs into a graph state dict in-place."""
        zero = torch.zeros((), dtype=_CDTYPE)
        for slot in self.control_output_slots:
            key = self.ctrl_node_key(sm_key, slot)
            graph_state[key] = outputs.get(slot, zero)
