"""driver_node.py — DriverNode for the KPN graph solver.

A driver receives PerformanceAtoms (just like MetaVoiceNode) and forwards them
to every consumer attached to its score_out port, one copy per edge, ratio-scaled.
With non-zero VoiceSpread parameters, each incoming atom is exploded into
n_samples atoms drawn from Gaussian distributions over phase, amplitude, and
onset time.  The batch of exploded atoms is written to the voice's FIFO; the
voice synthesises all of them and sums them into one signal that feeds back to
the driver's tensor input.  The driver passes that accumulated signal through to
its tensor output for the mixer.

score_contract is identical in shape to MetaVoiceNode.score_contract so that
TorchComposerNode and ScoreSequencerNode send atoms to a driver without
modification.  needs_source_events=True requests that atoms carry the full
NoteEvent list from the generating schedule.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from graph_solver import TensorNode, _CDTYPE
from parametric_curve import (
    ParametricCurve,
    default_chirp,
    default_envelope,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-voice Gaussian spread
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VoiceSpread:
    """Gaussian spread applied when exploding one atom into n_samples atoms.

    Each incoming atom produces n_samples atoms, each independently sampled:
        phase_offset  ~ N(0,          phase_std)     radians added to phase_offset
        amplitude     ~ N(1,          amplitude_std) multiplied onto velocity
        onset_time    ~ N(onset,      time_std_s)    seconds, clamped to >= 0

    Velocities are scaled by 1/n_samples so the total energy matches one atom.
    When all std values are 0 or n_samples == 1, the fast copy path is used.
    """
    n_samples:     int   = 1    # atoms generated per input atom
    phase_std:     float = 0.0  # std dev of phase offset (radians)
    amplitude_std: float = 0.0  # std dev of amplitude multiplier (mean = 1.0)
    time_std_s:    float = 0.0  # std dev of onset time jitter (seconds)

    def is_identity(self) -> bool:
        return (
            self.n_samples <= 1
            and self.phase_std == 0.0
            and self.amplitude_std == 0.0
            and self.time_std_s == 0.0
        )


# ──────────────────────────────────────────────────────────────────────────────
# DriverNode
# ──────────────────────────────────────────────────────────────────────────────

class DriverNode(nn.Module):
    """Graph-resident driver: receives atoms, optionally explodes them via
    Gaussian spread, forwards batches to voice consumers, and passes the
    accumulated voice signal through to its tensor output.

    Parameters
    ----------
    key:
        Unique graph node key.
    envelope_curve / chirp_curve:
        Placed in score_contract for TorchComposerNode curve binding.
    groups / pages:
        Group/page routing strings, same as MetaVoiceNode.
    duration_s / release_tail_s:
        Passed to the sequencer for job sizing.
    ratios:
        Per-consumer amplitude scale factors (lazy-extended with 1.0).
    spreads:
        Per-consumer VoiceSpread instances (lazy-extended with identity spread).
    sample_rate:
        Render sample rate in Hz — needed to convert time_std_s to samples.
    layer:
        Graph layer string.
    """

    def __init__(
        self,
        key: str,
        *,
        envelope_curve: Optional[ParametricCurve] = None,
        chirp_curve: Optional[ParametricCurve] = None,
        groups: tuple = (),
        pages: tuple = (),
        duration_s: float = 1.0,
        release_tail_s: float = 0.08,
        ratios: Optional[list] = None,
        spreads: Optional[list] = None,
        sample_rate: float = 48_000.0,
        layer: str = "voice",
    ) -> None:
        super().__init__()
        self.key = str(key)
        self.envelope_curve: ParametricCurve = envelope_curve or default_envelope()
        self.chirp_curve: ParametricCurve = chirp_curve or default_chirp()
        self.groups: tuple = tuple(groups)
        self.pages: tuple = tuple(pages)
        self.duration_s: float = float(duration_s)
        self.release_tail_s: float = float(release_tail_s)
        self.ratios: list = list(ratios) if ratios else []
        self.spreads: list = list(spreads) if spreads else []
        self.sample_rate: float = float(sample_rate)
        self.layer: str = str(layer)

        self._pending_atoms: list = []
        self._fifo_bank: Optional[Any] = None
        self._fifo_slot_key: str = ""

        self.score_contract: dict = {
            "kind": "driver_score_consumer",
            "voice_key": self.key,
            "groups": self.groups,
            "pages": self.pages,
            "aggregation": "mask",
            "envelope_curve": self.envelope_curve,
            "chirp_curve": self.chirp_curve,
            "duration_s": self.duration_s,
            "release_tail_s": self.release_tail_s,
            "needs_source_events": True,
        }

        self.subscription_ports: dict = {}

    # ──────────────────────────────────────────────────────────────────────────

    def attach_fifo(self, bank: Any, slot_key: str) -> None:
        self._fifo_bank = bank
        self._fifo_slot_key = str(slot_key)

    def _drain_fifo(self) -> None:
        if (
            self._fifo_bank is not None
            and self._fifo_slot_key
            and self._fifo_bank.has_slot(self._fifo_slot_key)
        ):
            raw = self._fifo_bank.try_read(self._fifo_slot_key)
            if raw is not None:
                self._pending_atoms = raw if isinstance(raw, list) else [raw]

    def reset(self) -> None:
        self._pending_atoms = []

    # ──────────────────────────────────────────────────────────────────────────

    def _ratio_for(self, index: int) -> float:
        while len(self.ratios) <= index:
            self.ratios.append(1.0)
        return self.ratios[index]

    def _spread_for(self, index: int) -> VoiceSpread:
        while len(self.spreads) <= index:
            self.spreads.append(VoiceSpread())
        return self.spreads[index]

    def _explode(self, atom: Any, spread: VoiceSpread, ratio: float) -> list:
        """Return n_samples atoms drawn from Gaussian spread, velocity-normalised."""
        n = max(1, spread.n_samples)
        scale = ratio / n
        out = []
        sr = self.sample_rate
        for _ in range(n):
            delta_rad = random.gauss(0.0, spread.phase_std) if spread.phase_std else 0.0
            amp = random.gauss(1.0, spread.amplitude_std) if spread.amplitude_std else 1.0
            dt_s = random.gauss(0.0, spread.time_std_s) if spread.time_std_s else 0.0
            dt_samp = int(dt_s * sr)
            new_onset = max(0, atom.onset_sample + dt_samp)
            new_onset_s = max(0.0, atom.onset_time_s + dt_s)
            new_vel = atom.velocity * max(0.0, amp) * scale
            out.append(dataclasses.replace(
                atom,
                onset_sample=new_onset,
                onset_time_s=new_onset_s,
                velocity=new_vel,
                phase_delta=delta_rad,
            ))
        return out

    def _forward_atoms_to_consumers(self) -> None:
        """Explode and copy pending atoms to every score_out consumer FIFO."""
        if not self._pending_atoms or self._fifo_bank is None:
            return
        consumers: dict = (
            self.subscription_ports.get("score_out", {}).get("consumers", {})
        )
        for i, (voice_node_key, contract_list) in enumerate(consumers.items()):
            dst = contract_list[0] if contract_list else {}
            ratio = self._ratio_for(i)
            spread = self._spread_for(i)
            slot_key = str(
                dst.get("object_fifo_key")
                or f"{self.key}_{voice_node_key}_atoms"
            )
            if not self._fifo_bank.has_slot(slot_key):
                self._fifo_bank.claim_object(slot_key, fifo_size=16)

            if spread.is_identity():
                if ratio == 1.0:
                    batch = self._pending_atoms
                else:
                    batch = [
                        dataclasses.replace(a, velocity=a.velocity * ratio)
                        for a in self._pending_atoms
                    ]
            else:
                batch = []
                for atom in self._pending_atoms:
                    batch.extend(self._explode(atom, spread, ratio))

            self._fifo_bank.write_batch(slot_key, batch)

    # ──────────────────────────────────────────────────────────────────────────

    def build_node(self) -> TensorNode:
        node = self

        def _hook() -> None:
            node._pending_atoms = []
            node._drain_fifo()
            node._forward_atoms_to_consumers()
            node._pending_atoms = []

        def _transform(x: Tensor) -> Tensor:
            node._drain_fifo()
            node._forward_atoms_to_consumers()
            return x.to(_CDTYPE)

        return TensorNode(
            key=self.key,
            layer=self.layer,
            transform=_transform,
            fire_before_start=True,
            hook_before_start=_hook,
            analytic_module=self,
            subscription_ports=("score_in", "score_out"),
            subscription_contracts={
                "score_in": self.score_contract,
                "score_out": {
                    "kind": "score_producer",
                    "transport": "object_fifo",
                    "object_fifo_suffix": "atoms",
                },
            },
        )
