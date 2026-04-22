"""physical_edges.py — Physically-derived complex edge weights for routing graphs.

For analytic (complex) signals, sub-sample physical delays reduce to phase
rotations on the carrier.  Lossy media add attenuation.  Both are captured
by a single complex scalar per edge — exactly what W_inst[dst, src] already is.

This module computes those scalars from real physical parameters at build time.
Runtime cost: zero.  The physics is precomputed into the routing matrix.

Two edge models
---------------
WireParams
    Coaxial or twisted-pair cable segment.  Models propagation delay,
    skin-effect attenuation, and dielectric loss via the full
    transmission-line propagation constant γ = √((R+jωL)(G+jωC)).

BusParams
    Mixer bus / PCB backplane.  Propagation delay along the trace plus
    capacitive loading that grows with the number of bus participants.
    Each additional participant adds capacitance, degrading high-frequency
    content in a physically correct size-dependent way.

Usage
-----
    from physical_edges import wire_weight, bus_weight, apply_wire_edge
    import cmath

    # One 3-metre balanced cable at 440 Hz carrier
    omega_0 = 2 * math.pi * 440.0
    w = wire_weight(WireParams(length_m=3.0), omega_0)
    # w is a complex number: magnitude = attenuation, angle = phase shift

    # Apply directly to a RoutingGraph
    apply_wire_edge(graph, src="mic1", dst="preamp_in",
                    params=WireParams(length_m=3.0), omega_0=omega_0)

Physical constants used
-----------------------
All defaults model a good-quality balanced audio cable (Canare L-4E6S class):
  velocity_factor ≈ 0.66 (polyethylene dielectric)
  resistance_per_m ≈ 0.05 Ω/m (22 AWG copper pair)
  capacitance_per_m ≈ 100 pF/m (twisted pair, shielded)
  conductance_per_m ≈ 1e-9 S/m (very low leakage)

Mixer bus defaults model a professional console backplane (100mm trace):
  velocity_factor ≈ 0.5 (FR4 PCB dielectric)
  resistance_per_m ≈ 0.1 Ω/m (35 μm copper, 1mm trace)
  capacitance_per_node ≈ 15 pF (IC input pin + pad capacitance)
  source_impedance ≈ 75 Ω (typical bus driver)
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Optional

# Speed of light in vacuum (m/s)
_C = 2.998e8


# ═══════════════════════════════════════════════════════════════════════════════
# Physical parameter dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WireParams:
    """Physical parameters for one cable/wire segment.

    Attributes
    ----------
    length_m
        Physical length of the cable in metres.
    velocity_factor
        Fraction of the speed of light at which the signal propagates.
        Typical values: 0.66 (solid PE), 0.78 (foam PE), 0.97 (air-spaced).
    resistance_per_m
        Series resistance per metre (Ω/m).  Models skin-effect losses.
        Increases as √f in reality; this is the value at the carrier frequency.
    capacitance_per_m
        Shunt capacitance per metre (F/m).  Determines characteristic impedance
        together with inductance_per_m.
    inductance_per_m
        Series inductance per metre (H/m).  If None, derived from
        velocity_factor and capacitance_per_m via L = 1 / (v² C).
    conductance_per_m
        Shunt conductance per metre (S/m).  Models dielectric leakage.
        Very small for quality cables; negligible below RF frequencies.
    """
    length_m:           float = 1.0
    velocity_factor:    float = 0.66
    resistance_per_m:   float = 0.05     # Ω/m — 22 AWG copper
    capacitance_per_m:  float = 100e-12  # F/m — 100 pF/m typical audio cable
    inductance_per_m:   Optional[float] = None   # derived if None
    conductance_per_m:  float = 1e-9    # S/m — very low leakage

    def derived_inductance(self) -> float:
        """L = 1 / (v² · C)  from phase velocity and capacitance."""
        v = self.velocity_factor * _C
        return 1.0 / (v ** 2 * self.capacitance_per_m)

    def characteristic_impedance(self) -> float:
        """Z₀ = √(L/C)."""
        L = self.inductance_per_m if self.inductance_per_m is not None \
            else self.derived_inductance()
        return math.sqrt(L / self.capacitance_per_m)


@dataclass
class BusParams:
    """Physical parameters for a mixer bus / PCB backplane segment.

    Attributes
    ----------
    n_participants
        Number of nodes currently connected to the bus.  Each additional
        participant loads the bus with capacitance_per_node farads.
    trace_length_m
        Physical length of the bus trace or backplane conductor in metres.
    velocity_factor
        Signal propagation speed as a fraction of c.
        Typical PCB FR4: 0.45–0.55.
    resistance_per_m
        Trace resistance per metre at the carrier frequency (Ω/m).
    capacitance_per_node
        Input capacitance contributed by each connected participant (F).
        Includes IC pin capacitance, via capacitance, and pad capacitance.
    source_impedance
        Output impedance of the bus driver (Ω).  Determines how badly
        capacitive loading degrades high-frequency content.
    """
    n_participants:      int   = 1
    trace_length_m:      float = 0.1     # 100 mm trace — typical console bus
    velocity_factor:     float = 0.50
    resistance_per_m:    float = 0.1     # Ω/m — 35 μm copper, 1 mm width
    capacitance_per_node: float = 15e-12 # F — 15 pF per node
    source_impedance:    float = 75.0    # Ω


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer function computation
# ═══════════════════════════════════════════════════════════════════════════════

def wire_weight(params: WireParams, omega_0: float) -> complex:
    """Compute H = exp(−γ·L) for a cable segment at carrier ω₀.

    γ = √((R + jωL)(G + jωC))  — full transmission-line propagation constant.

    For the analytic signal, this is one complex scalar encoding:
      |H| = exp(−α·L)  — attenuation (skin-effect + dielectric loss)
      ∠H = −β·L        — phase delay (propagation time × carrier frequency)

    Returns a complex number on or inside the unit circle.
    |H| = 1 means lossless; |H| < 1 means lossy.
    """
    R = params.resistance_per_m
    G = params.conductance_per_m
    C = params.capacitance_per_m
    L = params.inductance_per_m if params.inductance_per_m is not None \
        else params.derived_inductance()
    length = params.length_m
    omega  = omega_0

    # Propagation constant γ = √((R + jωL)(G + jωC))
    Z = complex(R, omega * L)   # series impedance per metre
    Y = complex(G, omega * C)   # shunt admittance per metre
    gamma = cmath.sqrt(Z * Y)   # complex propagation constant

    # Transfer function H = exp(−γ·L)
    return cmath.exp(-gamma * length)


def bus_weight(params: BusParams, omega_0: float) -> complex:
    """Compute H_bus for a loaded mixer bus at carrier ω₀.

    Two effects:
      1. Propagation delay along the trace: phase rotation −ω₀·τ_trace.
      2. Capacitive loading from N participants: RC low-pass attenuation
         |H| = 1 / √(1 + (ω₀·R_src·C_total)²)

    Larger buses (more participants) have heavier capacitive loading,
    producing the physically correct size-dependent incoherence: the bus
    attenuates high-frequency content as it grows.
    """
    v          = params.velocity_factor * _C
    tau_trace  = params.trace_length_m / v
    C_total    = params.n_participants * params.capacitance_per_node
    RC         = params.source_impedance * C_total

    # Propagation phase
    phase = -omega_0 * tau_trace

    # Capacitive loading attenuation (1st-order RC)
    attenuation = 1.0 / math.sqrt(1.0 + (omega_0 * RC) ** 2)

    # Trace resistive loss (for completeness, negligible at audio frequencies)
    R_total    = params.resistance_per_m * params.trace_length_m
    resistive  = math.exp(-R_total / (2.0 * max(params.source_impedance, 1e-9)))

    return cmath.rect(attenuation * resistive, phase)


# ═══════════════════════════════════════════════════════════════════════════════
# RoutingGraph integration
# ═══════════════════════════════════════════════════════════════════════════════

def apply_wire_edge(
    graph,          # RoutingGraph
    src: str,
    dst: str,
    params: WireParams,
    omega_0: float,
    router_key: str = "",
) -> None:
    """Add a physically-modelled wire edge to a RoutingGraph.

    The complex weight H = exp(−γ·L) is decomposed into amplitude and phase
    and stored in a RoutingEdge with delay_s=0 (sub-sample delay is the phase).
    """
    from routing_engine import RoutingEdge
    H = wire_weight(params, omega_0)
    graph.edges.append(RoutingEdge(
        src_key    = src,
        dst_key    = dst,
        weight     = abs(H),
        angle_rad  = cmath.phase(H),
        delay_s    = 0.0,            # sub-sample: encoded in angle_rad
        router_key = router_key,
        edge_kind  = "signal",
    ))
    for k in (src, dst):
        if k not in graph.nodes:
            graph.add_node(k)


def apply_bus_edge(
    graph,
    src: str,
    dst: str,
    params: BusParams,
    omega_0: float,
    router_key: str = "",
) -> None:
    """Add a physically-modelled bus edge to a RoutingGraph."""
    from routing_engine import RoutingEdge
    H = bus_weight(params, omega_0)
    graph.edges.append(RoutingEdge(
        src_key    = src,
        dst_key    = dst,
        weight     = abs(H),
        angle_rad  = cmath.phase(H),
        delay_s    = 0.0,
        router_key = router_key,
        edge_kind  = "signal",
    ))
    for k in (src, dst):
        if k not in graph.nodes:
            graph.add_node(k)


# ═══════════════════════════════════════════════════════════════════════════════
# ICL — Cable and Bus Presets
#
# Named real-world hardware profiles for quick graph construction.
# All WireParams are given for 1 metre length; scale length_m as needed.
# All BusParams use n_participants=1 as the default starting point — set
# n_participants to your actual channel count before calling bus_weight().
#
# Physical sources
# ----------------
# Cable capacitance / resistance: manufacturer published data sheets.
# PCB trace resistance: 35 μm copper, trace width given per preset.
# PCB velocity factor: FR4 substrate, εr ≈ 4.5 → v ≈ c/√εr ≈ 0.47–0.53c.
# Bus capacitance per node: IC pin cap (≈5 pF) + via cap (≈3 pF) +
#   pad cap (≈7 pF) for through-hole or 1206 SMD on 2-layer FR4.
# ═══════════════════════════════════════════════════════════════════════════════

WIRE_PRESETS: dict = {

    # ── Professional balanced audio cables ────────────────────────────────

    "Canare_L4E6S": WireParams(
        # Star-quad balanced cable — industry standard studio/live.
        # 100 pF/m, 0.045 Ω/m per conductor, velocity ≈ 0.66c (PE insulation).
        length_m=1.0, velocity_factor=0.66,
        resistance_per_m=0.045, capacitance_per_m=100e-12,
        conductance_per_m=1e-9,
    ),
    "Canare_L4E6S_10m": WireParams(
        length_m=10.0, velocity_factor=0.66,
        resistance_per_m=0.045, capacitance_per_m=100e-12,
        conductance_per_m=1e-9,
    ),
    "Mogami_2534": WireParams(
        # Star-quad balanced — slightly higher capacitance, very low noise.
        # 130 pF/m, 0.048 Ω/m.
        length_m=1.0, velocity_factor=0.65,
        resistance_per_m=0.048, capacitance_per_m=130e-12,
        conductance_per_m=1e-9,
    ),
    "Mogami_2549": WireParams(
        # Mogami high-definition balanced — 54 pF/m (low capacitance variant).
        length_m=1.0, velocity_factor=0.68,
        resistance_per_m=0.040, capacitance_per_m=54e-12,
        conductance_per_m=8e-10,
    ),
    "Belden_8412": WireParams(
        # Classic microphone cable — 115 pF/m, 0.052 Ω/m.
        length_m=1.0, velocity_factor=0.66,
        resistance_per_m=0.052, capacitance_per_m=115e-12,
        conductance_per_m=1e-9,
    ),
    "Gotham_GAC2": WireParams(
        # Gotham Audio balanced — low capacitance (95 pF/m), high flexibility.
        length_m=1.0, velocity_factor=0.67,
        resistance_per_m=0.040, capacitance_per_m=95e-12,
        conductance_per_m=8e-10,
    ),
    "Gepco_SR": WireParams(
        # Gepco broadcast-grade balanced.  95 pF/m, 0.042 Ω/m.
        length_m=1.0, velocity_factor=0.67,
        resistance_per_m=0.042, capacitance_per_m=95e-12,
        conductance_per_m=9e-10,
    ),
    "Klotz_MY206": WireParams(
        # Klotz MY 206 balanced stage/studio.  110 pF/m.
        length_m=1.0, velocity_factor=0.66,
        resistance_per_m=0.047, capacitance_per_m=110e-12,
        conductance_per_m=1e-9,
    ),

    # ── Instrument cables (unbalanced / high-Z) ───────────────────────────

    "Canare_GS6": WireParams(
        # Standard instrument cable — 120 pF/m, 0.055 Ω/m.
        length_m=1.0, velocity_factor=0.64,
        resistance_per_m=0.055, capacitance_per_m=120e-12,
        conductance_per_m=1e-9,
    ),
    "Mogami_2524": WireParams(
        # Mogami guitar cable — 137 pF/m.  Higher capacitance colours tone.
        length_m=1.0, velocity_factor=0.64,
        resistance_per_m=0.058, capacitance_per_m=137e-12,
        conductance_per_m=1.2e-9,
    ),
    "vintage_coax_instrument": WireParams(
        # Generic vintage coaxial instrument cable.  High cap, typical 1970s.
        length_m=1.0, velocity_factor=0.60,
        resistance_per_m=0.070, capacitance_per_m=160e-12,
        conductance_per_m=2e-9,
    ),

    # ── Patchbay / console internal wiring ────────────────────────────────

    "studio_patchbay_300mm": WireParams(
        # Short normalled patchbay connection — 0.3 m.
        length_m=0.30, velocity_factor=0.70,
        resistance_per_m=0.030, capacitance_per_m=80e-12,
        conductance_per_m=5e-10,
    ),
    "console_internal_wiring": WireParams(
        # Typical console internal interconnect — 0.5 m.
        length_m=0.50, velocity_factor=0.70,
        resistance_per_m=0.025, capacitance_per_m=75e-12,
        conductance_per_m=5e-10,
    ),
    "eurorack_bus_ribbon": WireParams(
        # Eurorack CV ribbon cable, 30 cm.
        length_m=0.30, velocity_factor=0.65,
        resistance_per_m=0.100, capacitance_per_m=90e-12,
        conductance_per_m=1e-9,
    ),

    # ── Generic convenience presets ───────────────────────────────────────

    "short_balanced_1m": WireParams(
        length_m=1.0, velocity_factor=0.66,
        resistance_per_m=0.05, capacitance_per_m=100e-12,
    ),
    "medium_balanced_5m": WireParams(
        length_m=5.0, velocity_factor=0.66,
        resistance_per_m=0.05, capacitance_per_m=100e-12,
    ),
    "long_balanced_30m": WireParams(
        length_m=30.0, velocity_factor=0.66,
        resistance_per_m=0.05, capacitance_per_m=100e-12,
    ),
    "lossless_1m": WireParams(
        # Pure phase delay, no attenuation — useful for testing.
        length_m=1.0, velocity_factor=0.66,
        resistance_per_m=1e-9, capacitance_per_m=100e-12,
        conductance_per_m=1e-30,
    ),
}


BUS_PRESETS: dict = {

    # ── Professional console mixing buses ────────────────────────────────
    # Figures are approximations — exact PCB layouts are proprietary.
    # Derived from published specifications, service manuals, and
    # reverse-engineered schematics available in the public domain.

    "SSL_4000_stereo_bus": BusParams(
        # SSL 4000 series stereo mix bus — 32 ch configuration.
        # FR4 PCB backplane, long trace routing, heavy capacitive loading.
        n_participants=32, trace_length_m=0.28,
        velocity_factor=0.50, resistance_per_m=0.12,
        capacitance_per_node=20e-12, source_impedance=75.0,
    ),
    "SSL_4000_group_bus": BusParams(
        # SSL 4000 group bus — typically 8 channels.
        n_participants=8, trace_length_m=0.12,
        velocity_factor=0.50, resistance_per_m=0.10,
        capacitance_per_node=16e-12, source_impedance=75.0,
    ),
    "SSL_9000_stereo_bus": BusParams(
        # SSL 9000 — wider frame, heavier bus loading than 4000 series.
        n_participants=48, trace_length_m=0.35,
        velocity_factor=0.50, resistance_per_m=0.12,
        capacitance_per_node=18e-12, source_impedance=75.0,
    ),
    "Neve_8078_stereo_bus": BusParams(
        # Neve 8078 / 8068 mix bus — discrete component, shorter PCB runs.
        # Lower capacitive loading than SSL due to point-to-point wiring.
        n_participants=24, trace_length_m=0.22,
        velocity_factor=0.52, resistance_per_m=0.10,
        capacitance_per_node=15e-12, source_impedance=100.0,
    ),
    "Neve_8078_group_bus": BusParams(
        n_participants=8, trace_length_m=0.10,
        velocity_factor=0.52, resistance_per_m=0.08,
        capacitance_per_node=14e-12, source_impedance=100.0,
    ),
    "API_1604_bus": BusParams(
        # API 1604 / 2448 — lunchbox-style 2-bus, short runs.
        n_participants=16, trace_length_m=0.14,
        velocity_factor=0.48, resistance_per_m=0.08,
        capacitance_per_node=12e-12, source_impedance=50.0,
    ),
    "MCI_JH600_bus": BusParams(
        # MCI JH600 series — 24/48 track large-format console.
        n_participants=24, trace_length_m=0.24,
        velocity_factor=0.50, resistance_per_m=0.11,
        capacitance_per_node=18e-12, source_impedance=75.0,
    ),

    # ── Modular / semi-pro buses ──────────────────────────────────────────

    "eurorack_cv_bus": BusParams(
        # Eurorack power/CV distribution bus — short, low capacitance.
        n_participants=12, trace_length_m=0.08,
        velocity_factor=0.50, resistance_per_m=0.06,
        capacitance_per_node=10e-12, source_impedance=50.0,
    ),
    "500_series_bus": BusParams(
        # API 500-series lunchbox summing bus.
        n_participants=10, trace_length_m=0.10,
        velocity_factor=0.48, resistance_per_m=0.07,
        capacitance_per_node=12e-12, source_impedance=50.0,
    ),

    # ── Generic convenience presets ───────────────────────────────────────

    "generic_8ch_bus": BusParams(
        n_participants=8, trace_length_m=0.10,
        velocity_factor=0.50, resistance_per_m=0.10,
        capacitance_per_node=15e-12, source_impedance=75.0,
    ),
    "generic_16ch_bus": BusParams(
        n_participants=16, trace_length_m=0.18,
        velocity_factor=0.50, resistance_per_m=0.11,
        capacitance_per_node=16e-12, source_impedance=75.0,
    ),
    "generic_32ch_bus": BusParams(
        n_participants=32, trace_length_m=0.28,
        velocity_factor=0.50, resistance_per_m=0.12,
        capacitance_per_node=18e-12, source_impedance=75.0,
    ),
    "generic_48ch_bus": BusParams(
        n_participants=48, trace_length_m=0.38,
        velocity_factor=0.50, resistance_per_m=0.13,
        capacitance_per_node=20e-12, source_impedance=75.0,
    ),
}


def wire_weight_at_sr(
    params: WireParams,
    sr: float,
    carrier_ratio: float = 0.25,
) -> complex:
    """Convenience: compute wire weight at sr × carrier_ratio Hz.

    When the carrier frequency is not yet known, sr/4 is a reasonable
    broadband midpoint for audio cables.
    """
    omega_0 = 2.0 * math.pi * sr * carrier_ratio
    return wire_weight(params, omega_0)


def bus_weight_at_sr(
    params: BusParams,
    sr: float,
    carrier_ratio: float = 0.25,
) -> complex:
    omega_0 = 2.0 * math.pi * sr * carrier_ratio
    return bus_weight(params, omega_0)
