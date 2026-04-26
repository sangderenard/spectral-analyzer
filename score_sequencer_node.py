"""score_sequencer_node.py — KPN score producer node for the graph solver.

ScoreSequencerNode holds a SparseScoreTensor and, before the solver epoch
begins, converts it into PerformanceAtom batches and writes each batch into
a preclaimed ObjectFifoSlot.  Voice nodes drain those slots during synthesis.

Group-subscription routing
--------------------------
The sequencer decides *which events go to which voice* using group membership:

  - Each score event is tagged with a group index in ``PARAM_PATTERN`` and a
    page index in ``PARAM_PAGE``.
  - ``group_keys``  maps group-index → group-name  (e.g. ``{0: "melody"}``)
  - ``page_keys``   maps page-index  → page-name   (e.g. ``{0: "verse"}``)
  - Each subscribed voice declares its groups and pages in its score contract
    (``"groups": ("melody",)`` and ``"pages": ("verse",)``).
  - Before building atoms the hook calls ``slice_score_for_consumer`` to
    produce a masked view of the score containing only the events the voice
    is entitled to play.  This is the logical kernel of group subscription.
  - If a voice declares no groups *and* no pages it receives all events
    (backward-compatible default).

Wiring overview
---------------
1. Build a ScoreSequencerNode and call build_node() to get a TensorNode.
2. Add that TensorNode to GraphSolver together with the voice nodes.
3. Connect them with a TensorEdge that carries semantic_role="score".
4. After GraphSolver.__init__:
   - subscription_ports["score_out"] is live on the sequencer (set by
     GraphSolver.ensure_subscription_port).
   - consumers dict is populated inside that payload by
     GraphSolver._auto_form_contract_edges, keyed by edge.dst_key.
   - the object FIFO slot is already claimed in the graph's EdgeFifoBank.
5. Call GraphSolver.dispatch_before_start() — fires _hook.
6. Voices drain atoms via their attached FIFO slot during run_schedule.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from edge_fifo_bank import EdgeFifoBank
    from graph_solver import TensorNode
    from torch_composer_engine import SparseScoreTensor


class ScoreSequencerNode:
    """Graph-resident score producer with group-subscription routing.

    Parameters
    ----------
    key:
        Unique graph node key — must match the src_key of every score edge
        leaving this node.
    score:
        SparseScoreTensor to sequence.  May contain multiple batch rows
        (B > 1) and multiple pages (P > 1).
    sample_rate:
        Render sample rate in Hz.
    fifo_bank:
        The shared EdgeFifoBank.  GraphSolver preclaims one ObjectFifoSlot per
        object-FIFO contract edge before _hook execution.
    layer:
        Graph layer string passed to TensorNode (default "score").
    group_keys:
        Optional mapping from PARAM_PATTERN integer index to group name string.
        Used to route events to voices that declare matching ``score_groups``.
        Example: ``{0: "melody", 1: "bass"}``.
    page_keys:
        Optional mapping from PARAM_PAGE integer index to page name string.
        Used to route events to voices that declare matching ``score_pages``.
        Example: ``{0: "verse", 1: "chorus"}``.
    """

    def __init__(
        self,
        key: str,
        score: "SparseScoreTensor",
        *,
        sample_rate: float,
        fifo_bank: "EdgeFifoBank | None",
        layer: str = "score",
        group_keys: dict[int, str] | None = None,
        page_keys: dict[int, str] | None = None,
    ) -> None:
        self.key = key
        self.score = score
        self.sample_rate = float(sample_rate)
        self.fifo_bank = fifo_bank
        self.layer = layer
        self.group_keys: dict[int, str] = dict(group_keys or {})
        self.page_keys: dict[int, str] = dict(page_keys or {})
        # GraphSolver.ensure_subscription_port replaces this with a live
        # payload dict keyed by port name.
        self.subscription_ports: dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_wanted_pages(self, consumer_pages: frozenset[str]) -> "frozenset[int] | None":
        """Return the set of page indices this consumer subscribes to, or None (no filter)."""
        if not consumer_pages or not self.page_keys:
            return None
        return frozenset(idx for idx, name in self.page_keys.items() if name in consumer_pages)

    def _resolve_wanted_groups(self, consumer_groups: frozenset[str]) -> "frozenset[int] | None":
        """Return the set of group indices this consumer subscribes to, or None (no filter)."""
        if not consumer_groups or not self.group_keys:
            return None
        return frozenset(idx for idx, name in self.group_keys.items() if name in consumer_groups)

    # ──────────────────────────────────────────────────────────────────────────

    def build_node(self) -> "TensorNode":
        """Return a TensorNode wired to fire _hook before the solver epoch."""
        from graph_solver import TensorNode

        def _hook() -> None:
            from parametric_curve import default_chirp, default_envelope
            from torch_composer_engine import (
                envelope_jobs_from_sparse_score,
                performance_atoms_from_jobs,
                slice_score_for_consumer,
            )

            consumers: Mapping[str, list[dict]] = (
                self.subscription_ports.get("score_out", {}).get("consumers", {})
            )
            for voice_node_key, contract_list in consumers.items():
                dst_contract: dict[str, Any] = contract_list[0] if contract_list else {}
                voice_key      = str(dst_contract.get("voice_key", voice_node_key))
                env_curve      = dst_contract.get("envelope_curve") or default_envelope()
                chirp_curve    = dst_contract.get("chirp_curve") or default_chirp()
                release_tail_s = float(dst_contract.get("release_tail_s", 0.08))

                # ── Group-subscription routing ─────────────────────────────
                # Resolve which page and group indices this consumer is entitled
                # to receive, then produce a masked score view for them only.
                consumer_pages = frozenset(
                    str(x) for x in dst_contract.get("pages", ()) if str(x)
                )
                consumer_groups = frozenset(
                    str(x) for x in dst_contract.get("groups", ()) if str(x)
                )
                wanted_pages  = self._resolve_wanted_pages(consumer_pages)
                wanted_groups = self._resolve_wanted_groups(consumer_groups)

                consumer_score = slice_score_for_consumer(
                    self.score,
                    wanted_pages=wanted_pages,
                    wanted_groups=wanted_groups,
                )
                # ─────────────────────────────────────────────────────────────

                jobs = envelope_jobs_from_sparse_score(
                    consumer_score,
                    sample_rate=self.sample_rate,
                    release_tail_s=release_tail_s,
                )
                if jobs.job_count == 0:
                    continue

                n = jobs.job_count
                atoms = performance_atoms_from_jobs(
                    jobs,
                    envelope_curves=[env_curve] * n,
                    chirp_curves=[chirp_curve] * n,
                    voice_key=voice_key,
                    sample_rate=self.sample_rate,
                )

                slot_key = str(
                    dst_contract.get("object_fifo_key") or f"{self.key}_{voice_node_key}_atoms"
                )
                if not self.fifo_bank.has_slot(slot_key):
                    raise RuntimeError(
                        f"Object FIFO slot {slot_key!r} was not prepared by GraphSolver"
                    )
                self.fifo_bank.write_batch(slot_key, atoms)

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=None,
            fire_before_start=True,
            hook_before_start=_hook,
            analytic_module=self,
            subscription_ports=("score_out",),
            subscription_contracts={
                "score_out": {
                    "kind": "score_producer",
                    "transport": "object_fifo",
                    "object_fifo_suffix": "atoms",
                }
            },
        )
