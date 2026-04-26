"""instrument_node.py — InstrumentNode for the KPN graph solver.

An InstrumentNode sits between the score layer (TorchComposerNode) and the
driver layer (DriverNode).  It groups any number of drivers that share a
physical instrument body — e.g., all the string drivers on a single violin,
all the resonant layers of a piano section, or the pad drivers of a synth that
share a common filter body.

Responsibilities
----------------
1. **Score forwarding**
   Receives PerformanceAtoms (same score_contract shape as DriverNode) and
   copies them to every child driver's FIFO, optionally ratio-scaled per
   driver.

2. **Sympathetic resonance negotiation**
   When atoms arrive for driver i at frequency f, the instrument consults
   the precomputed coupling matrix (built from ``ResonatorString`` positions
   and harmonic proximity via ``resonator_core``) to decide whether driver j
   should also receive a sympathetically triggered copy.  Only pairs whose
   coupling magnitude exceeds ``sympathy_threshold`` produce injections; the
   injected atom velocity is ``source_velocity × |coupling[j, i]|``.

3. **Body resonance — cavity engine**
   The accumulated driver signal is fed into a real instrument body model
   built by ``sm_plugins.orchestral_resonance._build_body_scene``.  The
   ``body_type`` parameter selects the panel geometry:

     ``"string_plate"``  — ribs + top plate + back plate (violin/cello body)
     ``"reed_box"``      — cylindrical bore (clarinet / sax)
     ``"brass_bell"``    — tapered bore + flaring bell
     ``"drum_shell"``    — cylindrical shell + membrane
     ``"pipe_column"``   — open organ pipe / flute
     ``"voice_body"``    — vocal tract
     ``"direct"``        — pass-through (no body solve)

   Samples are buffered in chunks of ``body_chunk_size`` samples; on each
   full chunk ``render_cavity_scene_step`` is called and the aperture-
   pressure output is queued.  Individual samples are then drained per
   solver tick.  A ``CavityStreamState`` persists overlap/history across
   chunks to preserve body ring-down continuity.

   The aperture pressure IS the instrument's output signal — it captures
   how sound exits the body opening after panel reflection, diffusion, and
   aperture feedback.

Graph topology
--------------

    Composer ──score──▶ InstrumentNode ──score──▶ DriverNode[0]
                                      ╰──────────▶ DriverNode[1]
                                      ╰──────────▶ DriverNode[N]

    DriverNode[0] ──mix_source──▶ InstrumentNode ──mix_source──▶ Mixer
    DriverNode[1] ──mix_source──▶ InstrumentNode
    DriverNode[N] ──mix_source──▶ InstrumentNode

score_contract is identical in shape to DriverNode.score_contract so
TorchComposerNode and ScoreSequencerNode address an InstrumentNode exactly
as they would a DriverNode.

Usage example
-------------
    from resonator_core import ResonatorString, StringCouplingConfig
    from instrument_node import InstrumentNode

    strings = [
        ResonatorString(key="body_1", fundamental_hz=146.8, decay_s=1.8,
                        drive_gain=0.9, x=0.35, y=-0.12),
        ResonatorString(key="body_2", fundamental_hz=220.0, decay_s=1.6,
                        drive_gain=0.85, x=0.50, y=0.0),
        ResonatorString(key="body_3", fundamental_hz=293.7, decay_s=1.4,
                        drive_gain=0.80, x=0.65, y=0.12),
    ]
    coupling_cfg = StringCouplingConfig(base_strength=0.22, max_coupling=0.50)
    instr = InstrumentNode(
        "instr_body",
        driver_keys=["driver_body_1", "driver_body_2", "driver_body_3"],
        resonator_strings=strings,
        coupling_config=coupling_cfg,
        body_type="string_plate",
        body_chunk_size=64,
        sympathy_threshold=0.04,
        sympathy_velocity_floor=0.005,
        groups=("pad_body",),
        duration_s=SONG_DUR_S,
        release_tail_s=0.25,
        sample_rate=SR,
    )
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from cavity_engine import (
    CavityScene,
    CavityStreamState,
    flush_cavity_stream_state,
    init_cavity_stream_state,
    render_cavity_scene_step,
)
from graph_solver import TensorNode, _CDTYPE
from parametric_curve import ParametricCurve, default_chirp, default_envelope
from resonator_core import (
    ResonatorString,
    ResonatorSolveResult,
    StringCouplingConfig,
    _safe_decay,
    build_string_coupling_matrix,
)
from sm_plugins.orchestral_resonance import _build_body_scene


# ──────────────────────────────────────────────────────────────────────────────
# InstrumentNode
# ──────────────────────────────────────────────────────────────────────────────

def _make_sym_decay_envelope(decay_s: float, duration_s: float) -> ParametricCurve:
    """Exponential-decay envelope shaped by the resonating string's physical decay time."""
    c = ParametricCurve(name="sym_decay", v_lo=0.0, v_hi=1.0)
    dur = max(duration_s, 1e-6)
    t_half = min(0.45, 0.693 * decay_s / dur)   # half-life as fraction of note duration
    c.add_point(0.00, 1.0)
    c.add_point(t_half, 0.5)
    c.add_point(min(0.95, t_half * 4.0), 0.02)
    c.add_point(1.00, 0.0)
    return c


class InstrumentNode(nn.Module):
    """Graph-resident instrument: groups drivers that share a physical body.

    Parameters
    ----------
    key:
        Unique graph node key.
    driver_keys:
        Ordered list of DriverNode keys this instrument owns.  Atoms are
        forwarded to every driver in this order; sympathetic resonance is
        negotiated across all pairs within the list.
    resonator_strings:
        One ``ResonatorString`` per driver, describing that driver's coupling
        point on the instrument body.  ``fundamental_hz`` is the characteristic
        resonant pitch of that bridge/mount position (not the played pitch).
        ``x`` / ``y`` are normalised body coordinates used to compute spatial
        coupling decay.  When fewer strings than drivers are supplied, the
        remainder receive default strings tuned to 220 Hz.
    coupling_config:
        ``StringCouplingConfig`` passed to ``build_string_coupling_matrix()``.
        Controls harmonic- and distance-weighted cross-coupling between drivers.
    body_type:
        Instrument body geometry type.  Selects the panel builder from
        ``sm_plugins.orchestral_resonance``.  One of:
        ``"string_plate"``, ``"reed_box"``, ``"brass_bell"``, ``"drum_shell"``,
        ``"pipe_column"``, ``"voice_body"``, ``"direct"``.
        Default: ``"string_plate"``.
    body_jitter_seed:
        Integer seed for per-instrument body panel jitter.  Use different
        seeds for each instrument in an ensemble so they sound subtly
        different.  ``None`` disables jitter.
    sympathy_threshold:
        Minimum coupling magnitude ``|C[j, i]|`` required to inject a
        sympathetic atom.  Lower values → more cross-talk.  Default 0.05.
    sympathy_velocity_floor:
        Sympathetic atoms below this velocity (after coupling scaling) are
        silently dropped.  Default 0.005.
    envelope_curve / chirp_curve:
        Placed in score_contract for TorchComposerNode curve binding.
    groups / pages:
        Group / page routing strings, same as DriverNode.
    duration_s / release_tail_s:
        Passed to the sequencer for job sizing.
    ratios:
        Per-driver amplitude scale factors for atom forwarding.
        Lazy-extended with 1.0.
    sample_rate:
        Render sample rate in Hz.
    layer:
        Graph layer string.
    """

    def __init__(
        self,
        key: str,
        driver_keys: list[str],
        *,
        resonator_strings: Optional[list[ResonatorString]] = None,
        coupling_config: Optional[StringCouplingConfig] = None,
        body_type: str = "string_plate",
        body_jitter_seed: Optional[int] = None,
        sympathy_threshold: float = 0.05,
        sympathy_velocity_floor: float = 0.005,
        envelope_curve: Optional[ParametricCurve] = None,
        chirp_curve: Optional[ParametricCurve] = None,
        groups: tuple = (),
        pages: tuple = (),
        duration_s: float = 1.0,
        release_tail_s: float = 0.08,
        ratios: Optional[list] = None,
        sample_rate: float = 48_000.0,
        layer: str = "voice",
    ) -> None:
        super().__init__()
        self.key = str(key)
        self.driver_keys: list[str] = list(driver_keys)
        self.envelope_curve: ParametricCurve = envelope_curve or default_envelope()
        self.chirp_curve:    ParametricCurve = chirp_curve or default_chirp()
        self.groups:         tuple = tuple(groups)
        self.pages:          tuple = tuple(pages)
        self.duration_s:     float = float(duration_s)
        self.release_tail_s: float = float(release_tail_s)
        self.ratios:         list  = list(ratios) if ratios else []
        self.sample_rate:    float = float(sample_rate)
        self.layer:          str   = str(layer)
        self.sympathy_threshold:     float = float(sympathy_threshold)
        self.sympathy_velocity_floor: float = float(sympathy_velocity_floor)
        self.body_type: str = str(body_type)

        # ── Resonator strings: one per driver (extended with defaults if short)
        n = len(driver_keys)
        base_strings: list[ResonatorString] = list(resonator_strings or [])
        while len(base_strings) < n:
            idx = len(base_strings)
            base_strings.append(ResonatorString(
                key=f"{key}_string_{idx}",
                fundamental_hz=220.0,
                decay_s=1.0,
                drive_gain=1.0,
                x=idx / max(n - 1, 1),
                y=0.0,
            ))
        self.resonator_strings: list[ResonatorString] = base_strings[:n]

        # ── Coupling matrix — precomputed, shape (n, n), complex128
        self._coupling_config = coupling_config or StringCouplingConfig()
        self._coupling_matrix: np.ndarray = build_string_coupling_matrix(
            self.resonator_strings, self._coupling_config
        )

        # ── Body cavity scene (None for "direct" type → passthrough)
        jitter_rng = random.Random(body_jitter_seed) if body_jitter_seed is not None else None
        self._body_scene: Optional[CavityScene] = _build_body_scene(
            self.body_type, jitter_rng=jitter_rng
        )

        # ── Cavity streaming state (initialised lazily on first step)
        self._cavity_state: Optional[CavityStreamState] = None

        # ── FIFO plumbing (attached by build_node / network_materializer)
        self._pending_atoms: list = []
        self._fifo_bank: Optional[Any] = None
        self._fifo_slot_key: str = ""

        # ── Decay envelope cache: (string_index, duration_s_bucket) → ParametricCurve
        self._sym_decay_cache: dict = {}

        self.score_contract: dict = {
            "kind":               "driver_score_consumer",
            "voice_key":          self.key,
            "groups":             self.groups,
            "pages":              self.pages,
            "aggregation":        "mask",
            "envelope_curve":     self.envelope_curve,
            "chirp_curve":        self.chirp_curve,
            "duration_s":         self.duration_s,
            "release_tail_s":     self.release_tail_s,
            "needs_source_events": True,
        }

        self.subscription_ports: dict = {}

        # ── Diagnostics (reset alongside cavity state)
        self._diag: dict = self._fresh_diag()

    @staticmethod
    def _fresh_diag() -> dict:
        return {
            "steps":           0,       # _transform calls
            "in_energy":       0.0,     # sum of ||x_in||^2 per step
            "out_energy":      0.0,     # sum of ||x_out||^2 per step
            "nan_clips":       0,       # times NaN/Inf guard fired
            "peak_ratio":      0.0,     # max(out_energy/in_energy) per step
            "atoms_recv":      0,       # total primary atoms drained from FIFO
            "atoms_primary":   0,       # primary copies written to driver FIFOs
            "atoms_sym":       0,       # sympathetic copies injected
            "sym_vel_sum":     0.0,     # total velocity of sympathetic copies
            "primary_vel_sum": 0.0,     # total velocity of primary copies
        }

    def print_diagnostics(self) -> None:
        """Print a compact energy-conservation report after the solve."""
        d = self._diag
        steps    = max(d["steps"], 1)
        in_e     = d["in_energy"]
        out_e    = d["out_energy"]
        ratio    = out_e / max(in_e, 1e-30)
        sym_pct  = (100.0 * d["atoms_sym"] / max(d["atoms_primary"] + d["atoms_sym"], 1))
        vel_mult = d["sym_vel_sum"] / max(d["primary_vel_sum"], 1e-30)
        print(
            f"\n[InstrumentNode '{self.key}' diagnostics]\n"
            f"  Solver steps       : {steps}\n"
            f"  Body type          : {self.body_type}\n"
            f"  --- CAVITY ENERGY ---\n"
            f"  Total in  energy   : {in_e:.4e}\n"
            f"  Total out energy   : {out_e:.4e}\n"
            f"  out/in ratio       : {ratio:.4f}  ({'GAIN — unstable' if ratio > 1.0 else 'loss — ok'})\n"
            f"  Peak step ratio    : {d['peak_ratio']:.4f}\n"
            f"  NaN/Inf clips      : {d['nan_clips']}\n"
            f"  --- ATOM ROUTING ---\n"
            f"  Atoms received     : {d['atoms_recv']}\n"
            f"  Primary copies out : {d['atoms_primary']}  vel_sum={d['primary_vel_sum']:.4f}\n"
            f"  Sympathetic copies : {d['atoms_sym']} ({sym_pct:.1f}% of total)  vel_sum={d['sym_vel_sum']:.4f}\n"
            f"  Sym velocity mult  : {vel_mult:.4f}  (extra energy injected per primary unit)\n"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # FIFO plumbing
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
        self._cavity_state = None
        self._diag = self._fresh_diag()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ratio_for(self, index: int) -> float:
        while len(self.ratios) <= index:
            self.ratios.append(1.0)
        return self.ratios[index]

    def rebuild_coupling(self) -> None:
        """Recompute the coupling matrix after strings or config are mutated."""
        self._coupling_matrix = build_string_coupling_matrix(
            self.resonator_strings, self._coupling_config
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Atom forwarding with sympathetic injection
    # ──────────────────────────────────────────────────────────────────────────

    def _forward_atoms_to_drivers(self) -> None:
        """Copy pending atoms to every driver FIFO and inject sympathetic copies.

        Mirrors DriverNode._forward_atoms_to_consumers: iterates
        subscription_ports["score_out"]["consumers"] (populated by
        _auto_form_contract_edges) so FIFO slot keys are derived from the
        graph-assigned contracts, not from self.driver_keys directly.

        For each driver i:
          - Forward all pending atoms ratio-scaled to i's FIFO.
          - For each atom forwarded to i at frequency f, inspect the coupling
            matrix column i.  For every driver j where ``|C[j, i]|`` exceeds
            ``sympathy_threshold``, inject a sympathetic copy into j's FIFO
            with velocity scaled by ``|C[j, i]|``.  Atoms below
            ``sympathy_velocity_floor`` are dropped.
        """
        if not self._pending_atoms or self._fifo_bank is None:
            return

        consumers: dict = (
            self.subscription_ports.get("score_out", {}).get("consumers", {})
        )
        if not consumers:
            return

        driver_order: list[str] = list(consumers.keys())
        n = len(driver_order)

        # Resolve FIFO slot keys from graph-assigned contracts (same as DriverNode).
        slot_for: dict[str, str] = {}
        for driver_key, contract_list in consumers.items():
            dst = contract_list[0] if contract_list else {}
            slot_key = str(
                dst.get("object_fifo_key")
                or f"{self.key}_{driver_key}_atoms"
            )
            if not self._fifo_bank.has_slot(slot_key):
                self._fifo_bank.claim_object(slot_key, fifo_size=16)
            slot_for[driver_key] = slot_key

        # Build per-driver batches: primary + sympathetic
        batches: dict[str, list] = {k: [] for k in driver_order}
        _d = self._diag
        _d["atoms_recv"] += len(self._pending_atoms)

        for i, src_key in enumerate(driver_order):
            ratio = self._ratio_for(i)
            for atom in self._pending_atoms:
                # Primary copy for driver i
                scaled = (
                    atom if ratio == 1.0
                    else dataclasses.replace(atom, velocity=atom.velocity * ratio)
                )
                batches[src_key].append(scaled)
                _d["atoms_primary"] += 1
                _d["primary_vel_sum"] += float(getattr(atom, "velocity", 0.0)) * ratio

                # Sympathetic copies for every other driver j
                atom_hz = float(getattr(atom, "fundamental_hz", 0.0))
                if atom_hz <= 0.0 or i >= self._coupling_matrix.shape[1]:
                    continue

                for j, dst_key in enumerate(driver_order):
                    if j == i or j >= self._coupling_matrix.shape[0]:
                        continue
                    c_mag = abs(self._coupling_matrix[j, i])
                    if c_mag < self.sympathy_threshold:
                        continue
                    sym_vel = atom.velocity * ratio * c_mag
                    if sym_vel < self.sympathy_velocity_floor:
                        continue
                    c_phase = float(np.angle(self._coupling_matrix[j, i]))
                    _dcache_key = (j, round(float(atom.duration_s), 2))
                    if _dcache_key not in self._sym_decay_cache:
                        self._sym_decay_cache[_dcache_key] = _make_sym_decay_envelope(
                            self.resonator_strings[j].decay_s,
                            atom.duration_s,
                        )
                    sym_env = self._sym_decay_cache[_dcache_key]
                    batches[dst_key].append(
                        dataclasses.replace(
                            atom,
                            velocity=sym_vel,
                            phase_delta=float(getattr(atom, "phase_delta", 0.0)) + c_phase,
                            envelope_curve=sym_env,
                        )
                    )
                    _d["atoms_sym"] += 1
                    _d["sym_vel_sum"] += sym_vel

        for driver_key, batch in batches.items():
            if batch:
                self._fifo_bank.write_batch(slot_for[driver_key], batch)

    # ──────────────────────────────────────────────────────────────────────────
    # Body resonance — cavity-engine-backed, one step per solver tick
    # ──────────────────────────────────────────────────────────────────────────

    def _init_cavity_state(self) -> None:
        """Lazily initialise (or re-initialise) the cavity stream state."""
        if self._body_scene is None:
            return
        self._cavity_state = init_cavity_stream_state(
            self._body_scene,
            self.sample_rate,
            device="cpu",
        )

    def _body_resonance_step(self, x: Tensor) -> Tensor:
        """Run one cavity step on the incoming driver signal.

        ``x`` is the accumulated complex tensor from all driver ``mix_source``
        edges — whatever block size the solver delivers (1 sample for stride=1,
        N samples for stride=N).  It is reshaped to ``(1, T)`` and passed
        directly to ``render_cavity_scene_step``.

        The body scene has one receiver placed at the aperture opening, so
        ``step.chunk_output[0]`` is the aperture-coloured output signal for
        this block.  ``step.state`` carries the overlap accumulator forward —
        this IS the ring-down tail: it is added into every subsequent chunk's
        output by the streaming step, so the body continues ringing after the
        driver goes silent without any manual tail management here.

        For the ``"direct"`` body type (no scene) the input passes through.
        """
        x_c = x.to(_CDTYPE)

        if self._body_scene is None:
            return x_c

        if self._cavity_state is None:
            self._init_cavity_state()

        # (1, T) — one source (the body driver)
        T = x_c.numel()
        src = x_c.reshape(1, T)

        # ── Diag: input energy
        _d = self._diag
        _d["steps"] += 1
        in_e = float(x_c.abs().pow(2).sum())
        _d["in_energy"] += in_e

        step = render_cavity_scene_step(
            self._body_scene,
            src,
            sample_rate=self.sample_rate,
            state=self._cavity_state,
            device="cpu",
        )
        self._cavity_state = step.state

        # chunk_output shape: (n_receivers, T) — receiver is at the aperture.
        out = step.chunk_output[0].reshape(x_c.shape).to(_CDTYPE)
        # Guard against cavity instability producing NaN/Inf.
        if not torch.isfinite(out).all():
            out = torch.zeros_like(out)
            _d["nan_clips"] += 1

        # ── Diag: output energy and peak ratio
        out_e = float(out.abs().pow(2).sum())
        _d["out_energy"] += out_e
        step_ratio = out_e / max(in_e, 1e-30)
        if step_ratio > _d["peak_ratio"]:
            _d["peak_ratio"] = step_ratio

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Graph node construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_node(self) -> TensorNode:
        """Return the TensorNode to register with a GraphSolver.

        Atom forwarding runs in hook_before_start (dispatch_before_start phase)
        so that voice nodes find their FIFOs populated when they fire in the
        lateral group during run_schedule.  The node must be registered AFTER
        TorchComposerNode and BEFORE DriverNodes and voice nodes so that
        dispatch_before_start fires in the correct order:
            Composer → InstrumentNode → DriverNodes → (voice FIFOs ready)

        Body resonance runs inside _transform as before.  The _transform score-
        delivery path is kept as a no-op fallback: after the hook consumes the
        FIFO, _drain_fifo returns None and _forward_atoms_to_drivers returns early.
        """
        node = self

        def _hook() -> None:
            node.reset()
            node._drain_fifo()
            node._forward_atoms_to_drivers()
            node._pending_atoms = []

        def _transform(x: Tensor) -> Tensor:
            node._drain_fifo()
            node._forward_atoms_to_drivers()
            node._pending_atoms = []
            return node._body_resonance_step(x)

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
                    "kind":               "score_producer",
                    "transport":          "object_fifo",
                    "object_fifo_suffix": "atoms",
                },
            },
        )
