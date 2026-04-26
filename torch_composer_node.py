"""torch_composer_node.py — Full composition pipeline as a graph-solver node.

Bridges the existing composition engines (rhythm_tree, dynamics_engine,
improv_engine, analytic_score) into the KPN solver architecture.  The node
owns an AnalyticPatch that drives the same _build_rhythm_schedule path used
by AnalyticDriverViewer — every UI edit to the patch's rhythm / dynamics /
improv / piano-roll settings is automatically picked up at the next epoch
because _hook() runs _build_rhythm_schedule fresh each time.

Pipeline (per subscribed voice consumer, executed in _hook)
-----------------------------------------------------------
    AnalyticPatch  +  degrees  +  deg_pattern
    ──► _build_rhythm_schedule()           (analytic_score.py)
        • WarpCurve  (swing / pocket / rubato / meter)
        • BeatTree   (subdivided rhythm grid)
        • NoteStream (degree table + pattern + probabilities)
        • apply_dynamics()                 (dynamics_engine.py)
        • apply_improv()                   (improv_engine.py)
    ──► NoteSchedule
    ──► score_tensor_from_schedules()      (torch_composer_engine.py)
    ──► ScoreTensor
    ──► sparse_score_tensor_from_score()   (torch_composer_engine.py)
    ──► SparseScoreTensor
    ──► envelope_jobs_from_sparse_score()  (torch_composer_engine.py)
    ──► ScoreEnvelopeJobs
    ──► performance_atoms_from_jobs()      (torch_composer_engine.py)
    ──► list[PerformanceAtom]
    ──► fifo_bank.write_batch()

UI integration
--------------
PatchPanel from analytic_driver.py is the natural UI for this node — it
already reads and writes AnalyticPatch directly.  Wire it up via
torch_composer_panel.TorchComposerPanel, which is a thin PatchPanel
subclass that binds to the node's patch:

    node  = TorchComposerNode("seq", patch=my_patch, ...)
    panel = TorchComposerPanel()
    panel.set_node(node)

The piano roll (EditorCanvas) also works unchanged: the node's _hook
populates patch.resolved_notes so the piano roll reflects the generated
schedule.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from edge_fifo_bank import EdgeFifoBank
    from graph_solver import TensorNode


class TorchComposerNode:
    """Graph-resident composition producer that runs the full analytic-driver
    composition pipeline before each solver epoch.

    Parameters
    ----------
    key:
        Unique graph node key — must match the src_key of every score edge
        leaving this node.
    patch:
        AnalyticPatch driving rhythm, dynamics, and improv configuration.
        Shared with the UI panels (PatchPanel, EditorCanvas) — edits are
        reflected immediately at the next epoch because _hook re-runs the
        pipeline from scratch.
    bpm:
        Tempo in beats per minute.
    degrees_by_group:
        Mapping from group name → list[float] of Hz values.
        Use ``"all"`` for a single shared degree table across all voices.
        Keys are the same strings used in each voice node's
        ``score_contract["groups"]`` tuple.
    deg_pattern_by_group:
        Mapping from group name → list[int] pattern indices into the
        corresponding degree table.  Use ``"all"`` for a shared pattern.
    sample_rate:
        Render sample rate in Hz.
    fifo_bank:
        Shared EdgeFifoBank.  GraphSolver preclaims one ObjectFifoSlot per
        contract edge before _hook runs.
    group_keys:
        Optional mapping from PARAM_PATTERN integer index → group name.
        Mirrors the ScoreSequencerNode convention.
    page_keys:
        Optional mapping from PARAM_PAGE integer index → page name.
    layer:
        Graph layer string passed to TensorNode (default ``"score"``).
    """

    def __init__(
        self,
        key: str,
        *,
        patch: Any,
        bpm: float = 120.0,
        degrees_by_group: Optional[Dict[str, list]] = None,
        deg_pattern_by_group: Optional[Dict[str, list]] = None,
        sample_rate: float = 48_000.0,
        fifo_bank: "EdgeFifoBank | None" = None,
        group_keys: Optional[Dict[int, str]] = None,
        page_keys: Optional[Dict[int, str]] = None,
        layer: str = "score",
    ) -> None:
        self.key = key
        self.patch = patch
        self.bpm = float(bpm)
        self.degrees_by_group: Dict[str, list] = dict(degrees_by_group or {"all": [440.0]})
        self.deg_pattern_by_group: Dict[str, list] = dict(deg_pattern_by_group or {"all": [0]})
        self.sample_rate = float(sample_rate)
        self.fifo_bank = fifo_bank
        self.layer = layer
        self.group_keys: Dict[int, str] = dict(group_keys or {})
        self.page_keys: Dict[int, str] = dict(page_keys or {})
        # GraphSolver.ensure_subscription_port replaces this with a live
        # payload dict keyed by port name — same contract as ScoreSequencerNode.
        self.subscription_ports: Dict[str, Any] = {}
        # Populated by _hook after each composition pass.  Keys = slot_key strings.
        # Survives until the next recompose so the UI can serialize on demand.
        self.last_atoms: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # Group / page helpers (mirror ScoreSequencerNode)
    # ──────────────────────────────────────────────────────────────────────────

    def _degrees_for_group(self, group_name: str) -> list:
        return (
            self.degrees_by_group.get(group_name)
            or self.degrees_by_group.get("all", [440.0])
        )

    def _pattern_for_group(self, group_name: str) -> list:
        return (
            self.deg_pattern_by_group.get(group_name)
            or self.deg_pattern_by_group.get("all", [0])
        )

    def _resolve_wanted_pages(self, consumer_pages: frozenset) -> Optional[frozenset]:
        if not consumer_pages or not self.page_keys:
            return None
        return frozenset(idx for idx, name in self.page_keys.items() if name in consumer_pages)

    def _resolve_wanted_groups(self, consumer_groups: frozenset) -> Optional[frozenset]:
        if not consumer_groups or not self.group_keys:
            return None
        return frozenset(idx for idx, name in self.group_keys.items() if name in consumer_groups)

    # ──────────────────────────────────────────────────────────────────────────
    # Graph node construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_node(self) -> "TensorNode":
        """Return a TensorNode wired to fire _hook before the solver epoch."""
        from graph_solver import TensorNode

        node = self

        def _hook() -> None:
            from analytic_score import _build_rhythm_schedule
            from analytic_model import ResolvedNote
            from parametric_curve import default_chirp, default_envelope
            from torch_composer_engine import (
                envelope_jobs_from_sparse_score,
                performance_atoms_from_jobs,
                score_tensor_from_schedules,
                sparse_score_tensor_from_score,
            )

            beat_s = 60.0 / max(node.bpm, 1.0)

            consumers: Mapping[str, list] = (
                node.subscription_ports.get("score_out", {}).get("consumers", {})
            )

            resolved: list = []
            node.last_atoms = []   # reset for this composition pass

            for voice_node_key, contract_list in consumers.items():
                dst            = contract_list[0] if contract_list else {}
                voice_key      = str(dst.get("voice_key", voice_node_key))
                env_curve      = dst.get("envelope_curve") or default_envelope()
                chirp_curve    = dst.get("chirp_curve") or default_chirp()
                release_tail_s = float(dst.get("release_tail_s", 0.08))
                consumer_groups = frozenset(
                    str(x) for x in dst.get("groups", ()) if str(x)
                )

                # Use the first declared group to select the patch page and
                # degree table.  "all" is the fallback when no group declared.
                group_name  = next(iter(consumer_groups), "all")
                degrees     = node._degrees_for_group(group_name)
                deg_pattern = node._pattern_for_group(group_name)

                # Fetch rhythm page, dynamics, and improv programs from the
                # patch using the same page-routing logic as the analytic driver.
                page     = node.patch.page_for(group_name)
                dyn_prog = node.patch.dynamics_for(group_name)
                imp_prog = node.patch.improv_for(group_name)

                # ── Full composition pipeline ─────────────────────────────────
                sched = _build_rhythm_schedule(
                    node.patch,
                    beat_s,
                    degrees,
                    deg_pattern,
                    page=page,
                    dynamics_program=dyn_prog,
                    improv_program=imp_prog,
                )

                if not sched.events:
                    continue

                # Collect events for piano roll before tensor conversion.
                for ev in sched.events:
                    resolved.append(ResolvedNote(
                        note_id=(
                            f"{voice_key}:{group_name}:"
                            f"{round(float(ev.start_time), 6)}:"
                            f"{round(float(ev.duration_s), 6)}:"
                            f"{round(float(ev.fundamental_hz), 4)}"
                        ),
                        voice_key=voice_key,
                        voice_label=voice_key,
                        layer_key=group_name,
                        start_time=float(ev.start_time),
                        duration_s=float(ev.duration_s),
                        fundamental_hz=float(ev.fundamental_hz),
                        velocity=float(ev.velocity),
                        locked=False,
                    ))

                # NoteSchedule → ScoreTensor → SparseScoreTensor
                score_t = score_tensor_from_schedules([sched])
                sparse  = sparse_score_tensor_from_score(score_t)

                # SparseScoreTensor → ScoreEnvelopeJobs → list[PerformanceAtom]
                jobs = envelope_jobs_from_sparse_score(
                    sparse,
                    sample_rate=node.sample_rate,
                    release_tail_s=release_tail_s,
                )
                if jobs.job_count == 0:
                    continue

                n     = jobs.job_count
                # Pass the full NoteEvent list when the consumer declared
                # needs_source_events (e.g. a DriverNode that re-composes
                # per-voice atoms from the same events under its own rules).
                src_events = (
                    list(sched.events)
                    if dst.get("needs_source_events")
                    else None
                )
                atoms = performance_atoms_from_jobs(
                    jobs,
                    envelope_curves=[env_curve] * n,
                    chirp_curves=[chirp_curve] * n,
                    voice_key=voice_key,
                    sample_rate=node.sample_rate,
                    source_events=src_events,
                )

                slot_key = str(
                    dst.get("object_fifo_key")
                    or f"{node.key}_{voice_node_key}_atoms"
                )
                if not node.fifo_bank.has_slot(slot_key):
                    node.fifo_bank.claim_object(slot_key, fifo_size=16)
                node.fifo_bank.write_batch(slot_key, atoms)
                node.last_atoms.extend(atoms)

            # Push collected notes to the patch so EditorCanvas / piano roll
            # reflects the generated schedule.  Direct assignment bypasses
            # _sync_resolved_notes which requires patch.voices to be populated.
            resolved.sort(key=lambda n: (n.start_time, n.fundamental_hz, n.voice_key))
            node.patch.resolved_notes = resolved

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
