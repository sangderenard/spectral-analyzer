"""analog_nodes.py — Analog circuit elements as complex operations on analytic signals.

Observation
-----------
For an analytic signal z(t) = A(t)·exp(jφ(t)), every circuit element
expressible as an LTI network reduces to a single complex multiplication
at the carrier frequency ω₀:

    z_out = H(ω₀) · z_in

H(ω₀) is precomputed once from physical parameters.  This covers:
  RC / RL / RLC filters, transmission line segments, transformer coupling,
  amplifiers with frequency-dependent response, cable loading, input
  impedances, mixer sum buses — everything linear.

Nonlinear elements operate on the complex envelope:
    z_out = f(|z_in|) · exp(j·arg(z_in))      # memoryless envelope nonlinearity
    z_out = f(z_in)                             # general complex nonlinearity

These become node_transforms in GraphSolver.from_graph().

Noise injection adds a complex perturbation:
    z_out = z_in + n_complex

Two usage patterns
------------------
Linear elements  → use the weight functions (rc_weight, rlc_weight, etc.)
    to compute a complex edge weight and call apply_*_edge from physical_edges,
    or insert directly via graph.set_weight / graph.set_angle_rad.

Nonlinear elements → instantiate a node class and pass it as a node_transform.

All callables accept and return torch.complex128 tensors (scalar or batched).
"""
from __future__ import annotations

import cmath
import math
from typing import Optional

import torch
from torch import Tensor

_CDTYPE = torch.complex128
_EPS    = 1e-30   # magnitude floor to avoid division by zero


# ═══════════════════════════════════════════════════════════════════════════════
# Linear element weights  (edge coefficients, precomputed at build time)
# ═══════════════════════════════════════════════════════════════════════════════

def rc_lowpass_weight(R: float, C: float, omega_0: float) -> complex:
    """H(ω₀) = 1 / (1 + jω₀RC)  — first-order RC low-pass."""
    return 1.0 / complex(1.0, omega_0 * R * C)


def rc_highpass_weight(R: float, C: float, omega_0: float) -> complex:
    """H(ω₀) = jω₀RC / (1 + jω₀RC)  — first-order RC high-pass."""
    jRC = complex(0.0, omega_0 * R * C)
    return jRC / (1.0 + jRC)


def rl_lowpass_weight(R: float, L: float, omega_0: float) -> complex:
    """H(ω₀) = R / (R + jω₀L)  — RL low-pass."""
    return complex(R, 0.0) / complex(R, omega_0 * L)


def rlc_bandpass_weight(R: float, L: float, C: float, omega_0: float) -> complex:
    """H(ω₀) = jω₀L / (R + jω₀L + 1/(jω₀C))  — series RLC bandpass."""
    jw   = complex(0.0, omega_0)
    Z    = complex(R, 0.0) + jw * L + 1.0 / (jw * C)
    return (jw * L) / Z


def rlc_resonator_weight(R: float, L: float, C: float, omega_0: float) -> complex:
    """Parallel RLC tank circuit H(ω₀).

    At resonance ω_r = 1/√(LC), |H| is maximised.  Off-resonance the
    impedance drops.  Models the frequency-selective loading of resonant
    structures (speaker crossovers, tone circuits, etc.).
    """
    jw  = complex(0.0, omega_0)
    Y   = 1.0 / complex(R, 0.0) + 1.0 / (jw * L) + jw * C
    return 1.0 / (Y * R)   # normalised to R at DC


def transformer_weight(
    turns_ratio:  float,
    leakage_L:    float,
    omega_0:      float,
    core_loss_R:  float = 1e6,
) -> complex:
    """Ideal transformer with leakage inductance and core loss.

    turns_ratio   = N_secondary / N_primary
    leakage_L     = total leakage inductance referred to primary (H)
    core_loss_R   = equivalent parallel core-loss resistance (Ω)
    """
    jw     = complex(0.0, omega_0)
    Z_leak = jw * leakage_L
    H_core = complex(core_loss_R, 0.0) / (complex(core_loss_R, 0.0) + Z_leak)
    return turns_ratio * H_core


def amplifier_weight(
    gain_db:      float,
    phase_deg:    float = 0.0,
    bandwidth_hz: Optional[float] = None,
    omega_0:      float = 0.0,
) -> complex:
    """Linear amplifier: gain in dB + phase shift.

    If bandwidth_hz is given, applies a first-order roll-off above that
    frequency: effectively an amplifier with a single-pole output filter.
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    if bandwidth_hz is not None and omega_0 > 0.0:
        omega_bw    = 2.0 * math.pi * bandwidth_hz
        pole_factor = 1.0 / complex(1.0, omega_0 / omega_bw)
        gain_linear *= abs(pole_factor)
        phase_deg   += math.degrees(cmath.phase(pole_factor))
    return cmath.rect(gain_linear, math.radians(phase_deg))


def capacitive_coupling_weight(
    C_coupling:   float,
    Z_load:       float,
    omega_0:      float,
) -> complex:
    """Capacitive crosstalk coupling coefficient.

    Models a coupling capacitor C_coupling between two conductors with
    load impedance Z_load on the victim side.
    """
    jw  = complex(0.0, omega_0)
    Zc  = 1.0 / (jw * C_coupling)
    return complex(Z_load, 0.0) / (complex(Z_load, 0.0) + Zc)


def inductive_coupling_weight(
    M:       float,   # mutual inductance (H)
    Z_load:  float,   # load impedance on victim (Ω)
    omega_0: float,
) -> complex:
    """Inductive (transformer-style) crosstalk at carrier frequency."""
    jw   = complex(0.0, omega_0)
    EMF  = jw * M                  # induced voltage per unit source current
    return EMF / complex(Z_load, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Nonlinear node transforms  (node_transforms callables for GraphSolver)
# ═══════════════════════════════════════════════════════════════════════════════

class TubeSaturation:
    """Triode/pentode soft saturation on the complex envelope.

    Models the plate-current nonlinearity of a vacuum tube via a
    soft-clipping function on the analytic signal's magnitude, preserving
    the instantaneous phase exactly (no phase-dependent crossover distortion).

    Transfer characteristic (envelope domain):
        A_out = gain × tanh(A_in / knee) × knee

    At small signals (A_in << knee): A_out ≈ gain × A_in  (linear region)
    At large signals (A_in >> knee): A_out → gain × knee   (saturation)

    Even-order harmonics are generated by asymmetry (bias parameter).
    bias > 0 shifts the operating point, breaking the odd-symmetry of tanh
    and introducing 2nd-harmonic content proportional to the bias level.
    """

    def __init__(
        self,
        gain:  float = 1.0,   # small-signal gain
        knee:  float = 1.0,   # saturation onset amplitude
        bias:  float = 0.0,   # DC bias (even-harmonic generator, 0=symmetric)
    ) -> None:
        self.gain = gain
        self.knee = knee
        self.bias = bias

    def __call__(self, z: Tensor) -> Tensor:
        mag   = z.abs().clamp(min=_EPS)
        A_in  = mag + self.bias
        A_out = self.gain * self.knee * torch.tanh(A_in / self.knee)
        return z * (A_out / mag)


class DiodeClip:
    """Hard-knee diode clipping on the complex envelope.

    Models a symmetric diode clipper (e.g. input protection, overdrive pedal).
    Above the knee the magnitude is compressed via a soft exponential rather
    than hard truncation, matching measured diode soft-knee behaviour.

    Transfer:
        A_out = knee × (1 − exp(−A_in / knee))
    """

    def __init__(self, knee: float = 1.0, gain: float = 1.0) -> None:
        self.knee = knee
        self.gain = gain

    def __call__(self, z: Tensor) -> Tensor:
        mag   = z.abs().clamp(min=_EPS)
        A_out = self.gain * self.knee * (1.0 - torch.exp(-mag / self.knee))
        return z * (A_out / mag)


class OpAmpClipper:
    """Rail-limited op-amp output stage.

    Linear below ±rail, hard-clips above.  Phase-preserving: the complex
    signal's direction is preserved, only the magnitude is clipped.
    """

    def __init__(self, rail: float = 1.0, gain: float = 1.0) -> None:
        self.rail = rail
        self.gain = gain

    def __call__(self, z: Tensor) -> Tensor:
        mag   = z.abs().clamp(min=_EPS)
        A_out = (self.gain * mag).clamp(max=self.rail)
        return z * (A_out / mag)


class HarmonicDistortion:
    """Polynomial harmonic generator on the complex envelope.

    Adds controlled harmonic content:
        z_out = z_in + h2 × z_in²/|z_in| + h3 × z_in³/|z_in|²

    The normalisation by |z_in|^(n-1) keeps the nth harmonic term at the
    same phase rotation as the fundamental to the nth power, which is the
    physically correct phasor relationship.

    h2 controls even-harmonic content (2nd, transformer saturation, asymmetric).
    h3 controls odd-harmonic content (3rd, tube triode, symmetric saturation).
    """

    def __init__(self, h2: float = 0.0, h3: float = 0.0) -> None:
        self.h2 = h2
        self.h3 = h3

    def __call__(self, z: Tensor) -> Tensor:
        mag = z.abs().clamp(min=_EPS)
        out = z.clone()
        if self.h2 != 0.0:
            out = out + self.h2 * (z * z) / mag
        if self.h3 != 0.0:
            out = out + self.h3 * (z * z * z) / (mag * mag)
        return out


class TransformerCore:
    """Magnetic core saturation for transformers and inductors.

    Combines linear coupling (turns ratio + leakage) with a nonlinear
    B-H curve approximated by the Langevin function:
        M(H) = M_sat × (coth(H/a) − a/H)

    For the analytic signal, H ≈ |z| (field proportional to signal amplitude)
    and the saturation manifests as envelope compression.
    """

    def __init__(
        self,
        turns_ratio: float = 1.0,
        M_sat:       float = 1.0,   # saturation magnetisation (normalised)
        a:           float = 0.3,   # Langevin shape parameter
    ) -> None:
        self.turns_ratio = turns_ratio
        self.M_sat       = M_sat
        self.a           = a

    def __call__(self, z: Tensor) -> Tensor:
        mag = z.abs().clamp(min=_EPS)
        x   = mag / self.a
        # Langevin function: L(x) = coth(x) - 1/x
        # Numerically stable: for large x, coth(x)≈1, so L(x)≈1-1/x
        langevin = torch.where(
            x > 20.0,
            1.0 - 1.0 / x.clamp(min=_EPS),
            (1.0 / torch.tanh(x.clamp(min=_EPS))) - (1.0 / x.clamp(min=_EPS)),
        )
        A_out = self.M_sat * langevin * self.turns_ratio
        return z * (A_out / mag)


# ═══════════════════════════════════════════════════════════════════════════════
# Noise injection  (node_transforms that add physically-modelled noise)
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalNoise:
    """Johnson-Nyquist thermal noise floor for a resistor.

    sigma² = 4 k_B T R Δf
    where Δf = sr / 2 (Nyquist bandwidth at the given sample rate).

    The noise is complex (real + imaginary parts independently Gaussian),
    as thermal noise is broadband and the analytic signal captures both
    quadratures.
    """

    _K_B = 1.380649e-23  # Boltzmann constant (J/K)

    def __init__(
        self,
        R:    float,              # resistance (Ω)
        T:    float = 293.15,     # temperature (K) — 20°C
        sr:   float = 48_000.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        delta_f   = sr / 2.0
        variance  = 4.0 * self._K_B * T * R * delta_f
        self.sigma = math.sqrt(variance)
        self.device = device

    def __call__(self, z: Tensor) -> Tensor:
        n_re = torch.randn(1, dtype=torch.float64, device=z.device) * self.sigma
        n_im = torch.randn(1, dtype=torch.float64, device=z.device) * self.sigma
        return z + torch.complex(n_re, n_im).to(_CDTYPE).squeeze()


class GroundLoop:
    """Deterministic ground-loop interference (50 Hz or 60 Hz hum).

    Adds a coherent complex sinusoidal disturbance:
        n(t) = amplitude × exp(j × (ω_hum × t + φ))

    The phase φ is fixed at construction — each GroundLoop instance is
    a different ground path with its own phase relationship.
    t is the sample index passed in on each call.
    """

    def __init__(
        self,
        amplitude:   float = 1e-4,
        freq_hz:     float = 50.0,
        phase_rad:   float = 0.0,
        sr:          float = 48_000.0,
    ) -> None:
        self.amplitude = amplitude
        self.omega     = 2.0 * math.pi * freq_hz
        self.phase     = phase_rad
        self.sr        = sr
        self._t        = 0

    def __call__(self, z: Tensor) -> Tensor:
        t   = self._t / self.sr
        hum = self.amplitude * cmath.exp(1j * (self.omega * t + self.phase))
        self._t += 1
        return z + torch.tensor(hum, dtype=_CDTYPE, device=z.device)

    def reset(self) -> None:
        self._t = 0


class CrosstalkInjection:
    """Capacitive or inductive crosstalk from a neighbouring channel.

    This is a two-input node transform: it reads a victim signal z and a
    source signal z_aggressor, returning z + H_cross × z_aggressor.

    H_cross is precomputed from physical coupling parameters.  Use
    capacitive_coupling_weight() or inductive_coupling_weight() to derive it.

    Usage in GraphSolver: use a coupled_transforms entry rather than
    node_transforms, since this node reads two graph nodes.
    """

    def __init__(self, H_cross: complex) -> None:
        self._H = torch.tensor(H_cross, dtype=_CDTYPE)

    def __call__(self, z_victim: Tensor, z_aggressor: Tensor) -> Tensor:
        return z_victim + self._H.to(z_victim.device) * z_aggressor


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: build a node_transforms dict from a declarative spec
# ═══════════════════════════════════════════════════════════════════════════════

def build_transforms(spec: dict) -> dict:
    """Build a node_transforms dict from a declarative parameter spec.

    spec format:
        {
            "node_key": {
                "type": "tube_sat" | "diode_clip" | "opamp_clip" |
                         "harmonic" | "transformer" |
                         "thermal_noise" | "ground_loop",
                <type-specific kwargs>
            },
            ...
        }

    ICL shorthand: a string value is treated as a COMPONENTS key.
        {"preamp": "12AX7"}   ≡   {"preamp": COMPONENTS["12AX7"]}

    Returns {node_key: callable} for use in GraphSolver.from_graph().
    """
    _registry = {
        "tube_sat":      lambda kw: TubeSaturation(**kw),
        "diode_clip":    lambda kw: DiodeClip(**kw),
        "opamp_clip":    lambda kw: OpAmpClipper(**kw),
        "harmonic":      lambda kw: HarmonicDistortion(**kw),
        "transformer":   lambda kw: TransformerCore(**kw),
        "thermal_noise": lambda kw: ThermalNoise(**kw),
        "ground_loop":   lambda kw: GroundLoop(**kw),
    }
    result = {}
    for node_key, params in spec.items():
        if isinstance(params, str):
            params = COMPONENTS[params]
        params  = dict(params)
        kind    = params.pop("type")
        factory = _registry.get(kind)
        if factory is None:
            raise ValueError(f"Unknown analog node type: {kind!r}")
        result[node_key] = factory(params)
    return result


def from_component(name: str) -> object:
    """Instantiate a single transform callable from an ICL component name.

    Example::

        tube = from_component("12AX7")
        diode = from_component("1N4148")
        transforms = {"preamp": tube, "clip": diode}
    """
    return build_transforms({"_": name})["_"]


# ═══════════════════════════════════════════════════════════════════════════════
# ICL — Ideal Component Library
#
# A constant dictionary of named analog devices with measured/published
# parameters translated into the coordinate system used by the node classes
# in this module.
#
# Amplitude conventions
# ---------------------
# All amplitude-domain parameters (knee, rail, bias) use *normalized*
# units where 1.0 = the nominal operating level of the circuit stage.
# For a ±15 V op-amp stage: 1.0 ≡ 15 V (the supply rail), so rail=0.90
# means the op-amp swings to ±13.5 V before hard-clipping.
# For a tube preamplifier at typical plate supply: 1.0 ≡ the nominal
# input signal amplitude at which the stage is specified to operate
# linearly.  Scale by your system's operating point as needed.
#
# Gain is dimensionless (voltage gain in the linear region).
#
# Source references
# -----------------
# Tube parameters: Mullard, RCA, Sylvania data sheets; Merlin Blencowe
#   "Designing Valve Preamps" (Crowood, 2009).
# Diode Vf: JEDEC/manufacturer data sheets at 1 mA forward current.
# Op-amp output swing: manufacturer data sheets at rated supply voltage.
# Transformer core: Lundahl/Jensen/Carnhill published specs + common
#   studio folklore calibrated against measured harmonic profiles.
# Harmonic profiles: IEC 60268-3 THD+N measurement methodology applied
#   to published measurements by Hamm (1973), Temme (1992), and Brandt
#   and Bohn "Audio Electronics" (Focal, 2001).
# Cable specs: manufacturer published data sheets.
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENTS: dict = {

    # ── Vacuum triodes / pentodes (TubeSaturation) ────────────────────────
    # gain  = small-signal voltage gain in the linear region.
    # knee  = saturation onset in normalized amplitude.
    # bias  = DC operating-point shift — positive → 2nd-harmonic asymmetry.
    #         Class A biasing: bias ≈ 0.  Class AB: bias 0.05–0.15.

    # High-mu triodes (preamp voltage gain stages)
    "12AX7": {
        # ECC83 / 7025 — the defining preamp triode.  mu ≈ 100.
        # Plate supply ≈ 250–300 V; cathode bias sets class-A operating point.
        "type": "tube_sat", "gain": 60.0, "knee": 0.15, "bias": 0.02,
    },
    "12AT7": {
        # ECC81 — medium-high mu (≈55), used in phase splitter / driver stages.
        "type": "tube_sat", "gain": 38.0, "knee": 0.25, "bias": 0.015,
    },
    "12AY7": {
        # Fender tweed-era preamp triode, lower mu (≈44) than 12AX7.
        "type": "tube_sat", "gain": 30.0, "knee": 0.30, "bias": 0.015,
    },

    # Medium-mu triodes (driver / phase-splitter / cathode follower)
    "12AU7": {
        # ECC82 — medium mu (≈17), high current, line-stage / driver.
        "type": "tube_sat", "gain": 17.0, "knee": 0.40, "bias": 0.010,
    },
    "5751": {
        # Low-noise 12AX7 substitute, mu ≈ 70.  Slightly softer saturation.
        "type": "tube_sat", "gain": 45.0, "knee": 0.18, "bias": 0.018,
    },
    "6SL7": {
        # Octal high-mu triode (mu ≈ 70), vintage preamplifiers.
        "type": "tube_sat", "gain": 42.0, "knee": 0.22, "bias": 0.02,
    },
    "6SN7": {
        # Octal medium-mu triode (mu ≈ 20), high headroom line stage / driver.
        "type": "tube_sat", "gain": 20.0, "knee": 0.55, "bias": 0.010,
    },

    # Power triodes (output stages — normalized to unity gain power stage context)
    "300B": {
        # Western Electric 300B — canonical SET power triode.  mu ≈ 3.85.
        # Warm, predominantly 2nd harmonic, very high headroom.
        "type": "tube_sat", "gain": 3.5, "knee": 0.85, "bias": 0.040,
    },
    "2A3": {
        # RCA 2A3 — lower-power cousin of 300B.  mu ≈ 4.2.
        "type": "tube_sat", "gain": 3.0, "knee": 0.75, "bias": 0.035,
    },
    "45": {
        # Older triode power tube, very clean low-order saturation.
        "type": "tube_sat", "gain": 2.5, "knee": 0.70, "bias": 0.030,
    },
    "211": {
        # Thoriated-tungsten transmitting triode, extreme headroom.
        "type": "tube_sat", "gain": 4.2, "knee": 1.20, "bias": 0.025,
    },

    # Beam power tetrodes / pentodes (output stages)
    "6L6GC": {
        # RCA beam power tetrode — American push-pull power amp.
        # Moderate saturation onset, stronger odd-harmonic content than triodes.
        "type": "tube_sat", "gain": 1.0, "knee": 0.80, "bias": 0.070,
    },
    "6V6GT": {
        # Small American tetrode — Fender tweed / brown-panel sound.
        # Earlier saturation onset, characteristically smooth clip.
        "type": "tube_sat", "gain": 1.0, "knee": 0.60, "bias": 0.060,
    },
    "EL34": {
        # Mullard power pentode — British push-pull power amp (Marshall et al.).
        # Harder, more aggressive saturation than 6L6.
        "type": "tube_sat", "gain": 1.0, "knee": 0.70, "bias": 0.065,
    },
    "EL84": {
        # Philips small power pentode — Vox AC-series breakup character.
        # Early saturation, predominantly even harmonics.
        "type": "tube_sat", "gain": 1.0, "knee": 0.55, "bias": 0.080,
    },
    "KT88": {
        # Beam power tetrode — extended headroom vs. EL34, British HiFi.
        "type": "tube_sat", "gain": 1.0, "knee": 0.90, "bias": 0.055,
    },
    "KT66": {
        # Older British beam tetrode — rounder saturation than KT88.
        "type": "tube_sat", "gain": 1.0, "knee": 0.75, "bias": 0.060,
    },

    # ── Diodes (DiodeClip) ────────────────────────────────────────────────
    # knee = forward threshold in normalized amplitude (1.0 = nominal stage level).
    # Vf values from JEDEC data sheets at 1 mA / 25°C.

    # Silicon signal diodes
    "1N4148": {
        # Standard fast-switching silicon.  Vf ≈ 0.65 V.
        "type": "diode_clip", "knee": 0.65, "gain": 1.0,
    },
    "1N914": {
        # Electrically equivalent to 1N4148.  Vf ≈ 0.64 V.
        "type": "diode_clip", "knee": 0.64, "gain": 1.0,
    },
    "1N4001": {
        # Silicon rectifier — slightly higher Vf due to power rating.  0.70 V.
        "type": "diode_clip", "knee": 0.70, "gain": 1.0,
    },
    "LED_red": {
        # Red LED used as asymmetric clipper (Rangemaster-style).  Vf ≈ 1.7 V.
        "type": "diode_clip", "knee": 1.70, "gain": 1.0,
    },

    # Schottky diodes (low forward voltage)
    "BAT41": {
        # Schottky signal diode — overdrive asymmetric clipper.  Vf ≈ 0.30 V.
        "type": "diode_clip", "knee": 0.30, "gain": 1.0,
    },
    "1N5819": {
        # Schottky power rectifier.  Vf ≈ 0.27 V at 1 mA.
        "type": "diode_clip", "knee": 0.27, "gain": 1.0,
    },
    "BAT85": {
        # Low-capacitance Schottky.  Vf ≈ 0.32 V.
        "type": "diode_clip", "knee": 0.32, "gain": 1.0,
    },

    # Germanium point-contact diodes (vintage fuzz / treble booster)
    "OA47": {
        # Mullard germanium.  Vf ≈ 0.16 V, soft knee.
        "type": "diode_clip", "knee": 0.16, "gain": 1.0,
    },
    "OA81": {
        # Mullard germanium.  Vf ≈ 0.18 V.
        "type": "diode_clip", "knee": 0.18, "gain": 1.0,
    },
    "D9E": {
        # Soviet germanium point-contact.  Vf ≈ 0.17 V.  Vintage Russian fuzz.
        "type": "diode_clip", "knee": 0.17, "gain": 1.0,
    },
    "AA119": {
        # Mullard germanium — Dallas Rangemaster and clones.  Vf ≈ 0.20 V.
        "type": "diode_clip", "knee": 0.20, "gain": 1.0,
    },
    "MP38A": {
        # Soviet germanium transistor used as diode (collector–base).  ≈ 0.19 V.
        "type": "diode_clip", "knee": 0.19, "gain": 1.0,
    },

    # ── Op-amps (OpAmpClipper) ────────────────────────────────────────────
    # rail = max output swing normalized to supply voltage.
    # At ±15 V: a swing to ±13.5 V → rail = 13.5/15 = 0.90.
    # gain = closed-loop gain factor (unity by default; set per circuit).
    # Figures from manufacturer data sheets, output loaded at 2 kΩ / 25°C.

    "NE5532": {
        # Philips / TI dual low-noise BJT input.  The industry workhorse.
        # Output swing ≈ ±13.5 V at ±15 V supply.
        "type": "opamp_clip", "rail": 0.90, "gain": 1.0,
    },
    "NE5534": {
        # Single low-noise equivalent of NE5532.  Slightly higher headroom.
        "type": "opamp_clip", "rail": 0.91, "gain": 1.0,
    },
    "TL072": {
        # JFET input — Roland, Fender, Boss era.  Lower output swing.
        "type": "opamp_clip", "rail": 0.87, "gain": 1.0,
    },
    "TL071": {
        # Single TL072 equivalent.
        "type": "opamp_clip", "rail": 0.87, "gain": 1.0,
    },
    "RC4558": {
        # BJT input — Ibanez Tube Screamer era.  Lower headroom, soft clip.
        "type": "opamp_clip", "rail": 0.83, "gain": 1.0,
    },
    "RC4559": {
        # Dual RC4558.
        "type": "opamp_clip", "rail": 0.83, "gain": 1.0,
    },
    "LM741": {
        # Original general-purpose BJT op-amp.  Lower swing, soft output clip.
        "type": "opamp_clip", "rail": 0.80, "gain": 1.0,
    },
    "UA741": {
        # Fairchild / TI version of the 741.
        "type": "opamp_clip", "rail": 0.80, "gain": 1.0,
    },
    "LM318": {
        # High-speed BJT op-amp — MXR Distortion+ era.
        "type": "opamp_clip", "rail": 0.86, "gain": 1.0,
    },
    "OPA2134": {
        # Burr-Brown FET input — high headroom, very clean, HiFi grade.
        "type": "opamp_clip", "rail": 0.93, "gain": 1.0,
    },
    "OPA627": {
        # Burr-Brown FET precision — near-ideal, studio grade.
        "type": "opamp_clip", "rail": 0.94, "gain": 1.0,
    },
    "AD797": {
        # Analog Devices ultra-low-noise BJT — studio microphone preamp.
        "type": "opamp_clip", "rail": 0.93, "gain": 1.0,
    },
    "LM4562": {
        # National / TI high-performance dual — modern standard.
        "type": "opamp_clip", "rail": 0.95, "gain": 1.0,
    },
    "THAT1512": {
        # THAT Corporation instrumentation amp — balanced microphone preamp.
        "type": "opamp_clip", "rail": 0.94, "gain": 1.0,
    },

    # ── Transformer cores (TransformerCore) ──────────────────────────────
    # turns_ratio = N_sec / N_pri.
    # M_sat = saturation magnetisation (normalised to nominal flux density).
    # a = Langevin shape parameter — smaller = harder, earlier saturation.

    "Lundahl_LL1538": {
        # Swedish audio input transformer — neutral, wide bandwidth.
        # Very clean, high headroom, minimal harmonic contribution.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 1.20, "a": 0.28,
    },
    "Lundahl_LL1935": {
        # MC step-up transformer — very high turns ratio (1:10).
        "type": "transformer", "turns_ratio": 10.0, "M_sat": 0.60, "a": 0.35,
    },
    "Jensen_JT11P1": {
        # Jensen input transformer — clean, wide bandwidth, studio standard.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 1.10, "a": 0.32,
    },
    "Jensen_JT115K": {
        # Jensen line input — higher saturation at low frequencies.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 1.05, "a": 0.28,
    },
    "Neve_Carnhill_input": {
        # Carnhill VTB9045 (Neve 1073 variant) — classic British color.
        # Pronounced low-order harmonic saturation at high levels.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 0.95, "a": 0.20,
    },
    "Neve_Carnhill_output": {
        # Carnhill output transformer — more saturation, contributes warmth.
        "type": "transformer", "turns_ratio": 0.5, "M_sat": 0.88, "a": 0.17,
    },
    "API_input": {
        # API 2520 / 312 input transformer — tight, punchy transient response.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 1.05, "a": 0.24,
    },
    "Triad_A11J": {
        # Vintage US output transformer — vintage tape-machine warmth.
        "type": "transformer", "turns_ratio": 0.3, "M_sat": 0.80, "a": 0.15,
    },
    "UTC_A20": {
        # Vintage UTC broadcast input — heavy iron, early saturation.
        "type": "transformer", "turns_ratio": 1.0, "M_sat": 0.75, "a": 0.14,
    },
    "Cinemag_CMMI": {
        # Cinemag CMMI moving-iron step-up — moderate saturation.
        "type": "transformer", "turns_ratio": 4.0, "M_sat": 0.90, "a": 0.22,
    },

    # ── Harmonic distortion profiles (HarmonicDistortion) ────────────────
    # h2 = 2nd harmonic (even) coefficient — asymmetric, characteristic of
    #      magnetic saturation, tube asymmetry, class-A bias drift.
    # h3 = 3rd harmonic (odd) coefficient — symmetric, characteristic of
    #      transistor crossover, clamped triode, symmetric feedback loops.
    # Values calibrated to approximate published THD+N measurements at
    # nominal operating level (IEC 60268-3 methodology).

    "tape_low": {
        # Tape saturation, moderate recording level (+0 VU on 250 nWb/m).
        "type": "harmonic", "h2": 0.040, "h3": 0.008,
    },
    "tape_nominal": {
        # Tape saturation, +4 dBu nominal level on 320 nWb/m bias.
        "type": "harmonic", "h2": 0.080, "h3": 0.016,
    },
    "tape_high": {
        # Tape saturation, +8 dB over nominal — classic warm tape sound.
        "type": "harmonic", "h2": 0.140, "h3": 0.030,
    },
    "tape_hot": {
        # Tape fully saturated — extreme warm compression effect.
        "type": "harmonic", "h2": 0.220, "h3": 0.055,
    },
    "vinyl_cutting": {
        # RIAA lacquer cutting — predominantly 2nd harmonic from cutter head.
        "type": "harmonic", "h2": 0.025, "h3": 0.004,
    },
    "vinyl_playback": {
        # Vinyl playback combined (cutter + stylus + cartridge).
        "type": "harmonic", "h2": 0.030, "h3": 0.008,
    },
    "guitar_humbucker": {
        # Magnetic pickup — asymmetric flux coupling gives 2nd harmonic.
        "type": "harmonic", "h2": 0.035, "h3": 0.018,
    },
    "guitar_singlecoil": {
        # Single-coil: more symmetric, higher 3rd from coil geometry.
        "type": "harmonic", "h2": 0.015, "h3": 0.028,
    },
    "carbon_comp_resistor": {
        # Carbon composition resistor nonlinearity (Hamm 1973).
        "type": "harmonic", "h2": 0.0020, "h3": 0.0008,
    },
    "electrolytic_cap_small": {
        # Small electrolytic — dielectric absorption at low frequencies.
        "type": "harmonic", "h2": 0.006, "h3": 0.002,
    },
    "electrolytic_cap_large": {
        # Large electrolytic power supply cap — stronger dielectric effect.
        "type": "harmonic", "h2": 0.012, "h3": 0.004,
    },
    "wire_wound_resistor": {
        # Wire-wound resistor — inductive at high frequency, very clean otherwise.
        "type": "harmonic", "h2": 0.0005, "h3": 0.0002,
    },

    # ── Thermal noise sources (ThermalNoise) ─────────────────────────────
    # Resistance in Ω; temperature defaults to 293.15 K (20°C).
    # sr is injected at build time by build_transforms / ThermalNoise.__init__.

    "R_1k": {
        "type": "thermal_noise", "R": 1_000.0, "T": 293.15,
    },
    "R_10k": {
        # 10 kΩ — typical mixer input resistor.
        "type": "thermal_noise", "R": 10_000.0, "T": 293.15,
    },
    "R_47k": {
        # 47 kΩ — guitar amplifier grid stopper / volume pot.
        "type": "thermal_noise", "R": 47_000.0, "T": 293.15,
    },
    "R_100k": {
        # 100 kΩ — console fader pad resistor (significant noise contributor).
        "type": "thermal_noise", "R": 100_000.0, "T": 293.15,
    },

    # ── Ground loop interference (GroundLoop) ─────────────────────────────
    # amplitude is in normalized units; adjust to suit your signal level.

    "hum_50hz": {
        # European / international power mains at 50 Hz.
        "type": "ground_loop", "amplitude": 5e-5, "freq_hz": 50.0, "phase_rad": 0.0,
    },
    "hum_60hz": {
        # North American / Japanese power mains at 60 Hz.
        "type": "ground_loop", "amplitude": 5e-5, "freq_hz": 60.0, "phase_rad": 0.0,
    },
    "hum_100hz": {
        # European 2nd harmonic — typical unbalanced cable in EU studio.
        "type": "ground_loop", "amplitude": 2e-5, "freq_hz": 100.0, "phase_rad": 0.3,
    },
    "hum_120hz": {
        # North American 2nd harmonic — unbalanced cable, USA studio.
        "type": "ground_loop", "amplitude": 2e-5, "freq_hz": 120.0, "phase_rad": 0.3,
    },
    "hum_50hz_severe": {
        # Severe European ground loop — single-ended instrument to balanced board.
        "type": "ground_loop", "amplitude": 3e-4, "freq_hz": 50.0, "phase_rad": 0.0,
    },
    "hum_60hz_severe": {
        # Severe North American ground loop.
        "type": "ground_loop", "amplitude": 3e-4, "freq_hz": 60.0, "phase_rad": 0.0,
    },
}
