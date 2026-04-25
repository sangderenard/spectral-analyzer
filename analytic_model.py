#!/usr/bin/env python3
"""Data model for the analytic driver."""
from __future__ import annotations

from analytic_shared import *  # noqa: F401,F403
import analytic_shared as _analytic_shared
from graph_solver import BlockFaculty

globals().update({
    k: v for k, v in vars(_analytic_shared).items()
    if not (k.startswith('__') and k.endswith('__'))
})

# ---------------------------------------------------------------------------
# Sequence / demo UI presets
# ---------------------------------------------------------------------------
_SEQ_SCALE_NAMES: list[str] = list(MODAL_SCALES.keys()) or ["pentatonic_minor"]
_SEQ_CHORD_NAMES: list[str] = list(CHORD_PROGRESSIONS.keys()) or ["I_IV_V_I"]

_RHYTHM_DIVISIONS: list[int] = [4, 8, 12, 16, 24, 32]
_SEQ_PATTERN_PRESETS: list[tuple[str, list[int]]] = [
    ("triad",    [0, 2, 4]),
    ("4-note",   [0, 2, 4, 7]),
    ("triad\u2195", [0, 2, 4, 2]),
    ("up-4",     [0, 1, 2, 3]),
    ("up-8",     [0, 1, 2, 3, 4, 5, 6, 7]),
    ("down-4",   [3, 2, 1, 0]),
    ("zigzag",   [0, 3, 1, 4, 2, 5]),
    ("walk",     [0, 1, 2, 3, 4, 3, 2, 1]),
    ("cascade",  [0, 2, 4, 7, 4, 2]),
    ("skip",     [0, 4, 2, 6, 1, 5]),
]
_SEQ_PATTERN_NAMES: list[str] = [n for n, _ in _SEQ_PATTERN_PRESETS]
_SEQ_RUBATO_SHAPES: list[str] = ["off", "sine", "troughs", "slow_go", "go_slow"]
_SEQ_RUBATO_SCOPES: list[str] = ["bar", "phrase"]
_METER_IRRATIONAL_SNAPS: list[float] = sorted([
    math.sqrt(2.0),
    math.sqrt(3.0),
    math.sqrt(5.0),
    (1.0 + math.sqrt(5.0)) * 0.5,
    math.e,
    math.pi,
    math.tau,
])

# Roman numeral → 0-based scale degree (try longest match first)
_ROMAN_DEGREE: dict[str, int] = {
    "VII": 6, "VI": 5, "IV": 3, "III": 2,
    "V": 4, "II": 1, "I": 0,
    "vii": 6, "vi": 5, "iv": 3, "iii": 2,
    "v": 4, "ii": 1, "i": 0,
}


def _chord_to_degree(chord_str: str) -> int:
    """Return 0-based scale degree from a Roman-numeral chord symbol."""
    s = chord_str.lstrip("-")
    for rn in ("VII", "VI", "IV", "III", "V", "II", "I",
               "vii", "vi",  "iv", "iii", "v",  "ii", "i"):
        if s.startswith(rn):
            return _ROMAN_DEGREE[rn]
    return 0


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
WIN_W_DEFAULT = 1600
WIN_H_DEFAULT = 900
PANEL_W = 280
TOPBAR_H = 30       # status / title bar
MODEBAR_H = 32      # mode-tab row height
BOTTOM_H = 28       # status-bar height

CP_HIT_PX   = 9    # control-point hit radius
CP_DRAW_PX  = 6    # diamond half-size for drawing
LOOP_HIT_PX = 8    # loop-handle hit column width

# ---------------------------------------------------------------------------
# Modulate-able voice/LFO/module attributes available for ParamNode targets.
# Each entry is the dot-path used in param_overrides / _synthesize_voice.
# ---------------------------------------------------------------------------
_VOICE_PARAM_ATTRS: list[str] = [
    "freq_hz",
    "amplitude",
    "semitone_offset",
    "chirp.f_delta_start",
    "chirp.f_delta_end",
    "adsr.attack",
    "adsr.decay",
    "adsr.sustain",
    "adsr.release",
    "fm.depth_hz",
    "fm.depth_amp",
    "am.depth_hz",
    "am.depth_amp",
    "harmonic_brightness",
    "harmonic_warp_strength",
    "harmonic_count",
    "granular.grain_density_hz",
    "granular.grain_duration_s",
    "granular.grain_scatter",
    "granular.grain_pitch_scatter",
    "granular.grain_manifold_mix",
    "granular.grain_amplitude_jitter",
]


# PlotWidget inner margins (must mirror PlotWidget constants)
_PW_ML = 36
_PW_MR = 6
_PW_MT = 14
_PW_MB = 14

# Colour palette (dark theme)
_C_BG       = (14,  14,  18,  255)
_C_PANEL    = (22,  22,  28,  255)
_C_WAVE     = (0.30, 0.62, 1.00, 1.0)
_C_ENV      = (1.00, 0.40, 0.20, 1.0)
_C_CHIRP    = (0.70, 0.30, 1.00, 1.0)
_C_LOOP     = (0.24, 0.78, 0.40, 1.0)
_C_LOOP_FILL= (0.10, 0.40, 0.18, 0.25)
_C_CURSOR   = (1.00, 0.72, 0.20, 1.0)
_C_CP_FILL  = (1.00, 0.80, 0.30, 0.9)
_C_CP_BORD  = (1.00, 1.00, 1.00, 1.0)
_C_GRID     = (0.16, 0.16, 0.20, 0.5)
_C_TAB_ACT  = (0.27, 0.51, 0.78, 1.0)
_C_TAB_IDLE = (0.14, 0.14, 0.18, 1.0)
_C_TXT      = (200, 200, 205)

# Pygame RGBA colour versions of some of the above
_PY_BG   = (14, 14, 18)
_PY_TXT  = (200, 200, 205)
_PY_DIM  = (90, 90, 100)
_PY_ACT  = (70, 130, 200)
_PY_WARN = (220, 100, 50)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ADSRParams:
    attack:  float = 0.005
    decay:   float = 0.04
    sustain: float = 0.75
    release: float = 0.08
    peak:    float = 1.0

    def to_knots(self, duration: float = 1.0) -> list[list[float]]:
        """Return ADSR as normalized knots list[[t, v], …]."""
        d = max(duration, 1e-9)
        a  = min(self.attack,  d)
        dk = min(self.decay,   d - a)
        r  = min(self.release, d - a - dk)
        ts = a + dk
        te = max(d - r, ts + 0.001 * d)
        return [
            [0.0,         0.0],
            [a  / d,      self.peak],
            [ts / d,      self.sustain],
            [te / d,      self.sustain],
            [1.0,         0.0],
        ]


@dataclass
class ChirpSpec:
    f_delta_start: float = 0.0    # Hz deviation at t=0 (added to base)
    f_delta_end:   float = 0.0    # Hz deviation at t=duration
    chirp_type:    str   = "none" # "none" | "linear" | "exponential" | "power"
    tau:           float = 0.5    # exponential decay constant (s)
    chirp_power:   float = 1.0    # exponent for "power" chirp type

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _CT = ["none", "linear", "exponential", "power"]
        return [
            KnobSpec("chirp_type",    "Chirp type", "choice", "none", 0, 3, 1, "",   _CT, False, "Chirp", ".0f", "",                     True),
            KnobSpec("f_delta_start", "\u0394f start",  "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f", "LinearChirpPhasePath"),
            KnobSpec("f_delta_end",   "\u0394f end",    "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f"),
            KnobSpec("tau",           "Tau",        "float",  0.5,  0.01,   10.0,   0, "s",  [], True,  "Chirp", ".3f", "ExponentialDecayPhasePath"),
            KnobSpec("chirp_power",   "Power",      "float",  1.0,  0.1,    8.0,    0, "",   [], False, "Chirp", ".2f", "PowerLawDecayPhasePath"),
        ]


@dataclass
class ModRouting:
    source_key: str   = ""
    depth_hz:   float = 2.0   # FM depth (Hz deviation)
    depth_amp:  float = 0.20  # AM depth (fraction of amplitude)


@dataclass
class PiecewiseVoiceEnvelope:
    curve: ParametricCurve = field(default_factory=lambda: _pc_default_envelope("voice_piecewise_amp"))
    chirp_curve: ParametricCurve = field(default_factory=lambda: _pc_default_chirp("voice_piecewise_chirp"))
    signal_curve: ParametricCurve = field(default_factory=lambda: _pc_default_blank("voice_piecewise_signal"))
    rule_tree: EnvelopeRuleTree = field(default_factory=EnvelopeRuleTree.default)
    source_path: str = ""
    detected_envelope_path: str = ""

    def to_dict(self) -> dict:
        return {
            "curve": self.curve.to_dict(),
            "chirp_curve": self.chirp_curve.to_dict(),
            "signal_curve": self.signal_curve.to_dict(),
            "rule_tree": self.rule_tree.to_dict(),
            "source_path": self.source_path,
            "detected_envelope_path": self.detected_envelope_path,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "PiecewiseVoiceEnvelope | None":
        if not d:
            return None
        curve = ParametricCurve.from_dict(dict(d.get("curve", {}))) if d.get("curve") else _pc_default_envelope("voice_piecewise_amp")
        chirp_curve = ParametricCurve.from_dict(dict(d.get("chirp_curve", {}))) if d.get("chirp_curve") else _pc_default_chirp("voice_piecewise_chirp")
        signal_curve = ParametricCurve.from_dict(dict(d.get("signal_curve", {}))) if d.get("signal_curve") else _pc_default_blank("voice_piecewise_signal")
        rule_tree = EnvelopeRuleTree.from_dict(dict(d.get("rule_tree", {}))) if d.get("rule_tree") else EnvelopeRuleTree.default()
        return cls(
            curve=curve,
            chirp_curve=chirp_curve,
            signal_curve=signal_curve,
            rule_tree=rule_tree,
            source_path=str(d.get("source_path", "") or ""),
            detected_envelope_path=str(d.get("detected_envelope_path", "") or ""),
        )


@dataclass
class AnalyticVoice:
    ARCHETYPE_KEY: ClassVar[str] = "analytic.voice"

    key:          str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:        str   = "Voice"
    freq_hz:      float = 440.0
    semitone_offset: float = 0.0   # semitones added on top of the context pitch
    note_tracking:   str   = "note"  # "note" | "root" | "free"
    amplitude:    float = 1.0
    phase_origin: float = 0.0      # radians
    chirp:  ChirpSpec  = field(default_factory=ChirpSpec)
    fm:     Optional[ModRouting] = None
    am:     Optional[ModRouting] = None
    env_type: str = "piecewise"
    adsr:     ADSRParams = field(default_factory=ADSRParams)
    env_knots: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0], [0.01, 1.0], [0.1, 0.75], [0.85, 0.75], [1.0, 0.0]
    ])
    piecewise_env: Optional[PiecewiseVoiceEnvelope] = None
    loop_start:   float = 0.1
    loop_end:     float = 0.9
    loop_enabled: bool  = False
    muted:        bool  = False
    color: list[int] = field(default_factory=lambda: [100, 160, 255])
    pre_delay:           float = 0.0   # seconds of silence before voice onset
    manifold_type:       str   = "pure"  # "pure" | "harmonic" | "harmonic_warp"
    harmonic_count:      int   = 8       # number of harmonics to sum
    harmonic_brightness: float = 1.0    # amplitude rolloff exponent: amp_k = 1/k^brightness
    harmonic_warp_strength: float = 0.0  # stretches harmonic ratios; 0 = exact integer multiples
    voice_role:          str   = "signal"  # "signal" | "air" | "transient" | "body"
    seq_role:            str   = "melody"  # "melody" | "bass" | "root" | "stab"
    register:            str   = "all"     # "all" | "bass" | "mid" | "high" — rhythm page routing
    polyphony_count:     int   = 1         # how many simultaneous lines fit before another chair is needed
    polyphony_mode:      str   = "sympathetic"  # "sympathetic" | "unsympathetic"
    body_type:           str   = "direct"  # standard instrument body / resonator type tag
    emission_mode:       str   = "single"  # "single" | "granular"
    granular:            Any   = None      # GrainPopulationSpec when emission_mode=="granular"

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"freq_hz", "amplitude", "semitone_offset",
                                       "harmonic_count", "harmonic_brightness"}),
            tracked_inputs=frozenset({"fm_in", "am_in", "env_mod_in",
                                      "chirp_mod_in", "pitch_in"}),
        )

    def active_knots(self) -> list[list[float]]:
        if self.env_type == "adsr":
            return self.adsr.to_knots(1.0)
        if self.piecewise_env is not None:
            return [[float(p.t), float(p.v)] for p in self.piecewise_env.curve.points]
        return self.env_knots

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label,
            "freq_hz": self.freq_hz,
            "semitone_offset": self.semitone_offset,
            "note_tracking":   self.note_tracking,
            "amplitude": self.amplitude,
            "phase_origin": self.phase_origin,
            "chirp": {
                "f_delta_start": self.chirp.f_delta_start,
                "f_delta_end":   self.chirp.f_delta_end,
                "chirp_type":    self.chirp.chirp_type,
                "tau":           self.chirp.tau,
                "chirp_power":   self.chirp.chirp_power,
            },
            "fm": ({"source_key": self.fm.source_key,
                    "depth_hz":   self.fm.depth_hz,
                    "depth_amp":  self.fm.depth_amp} if self.fm else None),
            "am": ({"source_key": self.am.source_key,
                    "depth_hz":   self.am.depth_hz,
                    "depth_amp":  self.am.depth_amp} if self.am else None),
            "env_type": "piecewise",
            "adsr": {"attack": self.adsr.attack, "decay": self.adsr.decay,
                     "sustain": self.adsr.sustain, "release": self.adsr.release,
                     "peak": self.adsr.peak},
            "env_knots": self.env_knots,
            "piecewise_env": self.piecewise_env.to_dict() if self.piecewise_env is not None else None,
            "loop_start": self.loop_start, "loop_end": self.loop_end,
            "loop_enabled": self.loop_enabled, "muted": self.muted,
            "color": self.color,
            "pre_delay": self.pre_delay,
            "manifold_type": self.manifold_type,
            "harmonic_count": self.harmonic_count,
            "harmonic_brightness": self.harmonic_brightness,
            "harmonic_warp_strength": self.harmonic_warp_strength,
            "voice_role": self.voice_role,
            "seq_role":   self.seq_role,
            "register":   self.register,
            "polyphony_count": self.polyphony_count,
            "polyphony_mode":  self.polyphony_mode,
            "body_type": self.body_type,
            "emission_mode": self.emission_mode,
            "granular": (self.granular.to_dict()
                         if self.granular is not None and hasattr(self.granular, "to_dict")
                         else self.granular),
        }

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _ROLES    = ["signal", "air", "transient", "body"]
        _MANIFOLD = ["pure", "harmonic", "harmonic_warp"]
        _POLY_MODES = ["sympathetic", "unsympathetic"]
        _BODY_TYPES = ["direct", "string_plate", "reed_box", "brass_bell", "drum_shell", "pipe_column", "voice_body"]
        return [
            # Oscillator
            KnobSpec("freq_hz",       "Freq",       "float",  440.0, 1.0,    20000.0, 0, "Hz",  [], True,  "Oscillator", ".1f", "ConstantPhasePath"),
            KnobSpec("semitone_offset","Offset",   "float",  0.0, -48.0,   48.0,    0, "st",  [], False, "Oscillator", ".2f"),
            KnobSpec("note_tracking",  "Tracking", "choice", "note", 0, 2, 1, "",
                     ["note", "root", "free"], False, "Oscillator", "", "", True),
            KnobSpec("amplitude",    "Amplitude",  "float",  1.0,   0.0,     4.0,     0, "",    [], False, "Oscillator", ".3f"),
            KnobSpec("phase_origin", "Phase",      "float",  0.0,  -math.pi, math.pi, 0, "rad", [], False, "Oscillator", ".3f", "ConstantPhasePath"),
            KnobSpec("pre_delay",    "Pre-delay",  "float",  0.0,   0.0,     2.0,     0, "s",   [], False, "Oscillator", ".3f"),
            KnobSpec("voice_role",   "Role",       "choice", "signal", 0, 3, 1, "",   _ROLES,    False, "Oscillator"),
            KnobSpec("seq_role",     "Arr. role",  "choice", "melody", 0, 3, 1, "",
                     ["melody", "bass", "root", "stab"], False, "Oscillator"),
            KnobSpec("register",     "Register",   "choice", "all",    0, 3, 1, "",
                     ["all", "bass", "mid", "high"],     False, "Oscillator"),
            KnobSpec("polyphony_count", "Polyphony", "int", 1, 1, 16, 1, "", [], False, "Oscillator", ".0f"),
            KnobSpec("polyphony_mode",  "Poly mode", "choice", "sympathetic", 0, 1, 1, "",
                     _POLY_MODES, False, "Oscillator"),
            KnobSpec("body_type",       "Body",       "choice", "direct", 0, max(0, len(_BODY_TYPES) - 1), 1, "",
                     _BODY_TYPES, False, "Oscillator"),
            # Chirp — delegate to ChirpSpec's own knob list with path prefix
            KnobSpec("chirp.chirp_type",    "Chirp type", "choice", "none", 0, 3, 1, "", ["none","linear","exponential","power"], False, "Chirp", ".0f", "", True),
            KnobSpec("chirp.f_delta_start", "\u0394f start",   "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f", "LinearChirpPhasePath"),
            KnobSpec("chirp.f_delta_end",   "\u0394f end",     "float",  0.0, -5000.0, 5000.0, 0, "Hz", [], False, "Chirp", ".1f"),
            KnobSpec("chirp.tau",           "Tau",         "float",  0.5,  0.01,   10.0,   0, "s",  [], True,  "Chirp", ".3f", "ExponentialDecayPhasePath"),
            KnobSpec("chirp.chirp_power",   "Power",       "float",  1.0,  0.1,    8.0,    0, "",   [], False, "Chirp", ".2f", "PowerLawDecayPhasePath"),
            # FM
            KnobSpec("fm.depth_hz",  "FM depth",   "float", 0.0,  0.0, 2000.0, 0, "Hz", [], True,  "FM", ".1f", "SmoothSinusoidalDriftModel"),
            # AM
            KnobSpec("am.depth_amp", "AM depth",   "float", 0.0,  0.0, 1.0,    0, "",   [], False, "AM", ".3f"),
            # Loop
            KnobSpec("loop_start",   "Loop start", "float", 0.1,  0.0, 0.99,   0, "",   [], False, "Loop", ".3f"),
            KnobSpec("loop_end",     "Loop end",   "float", 0.9,  0.01, 1.0,   0, "",   [], False, "Loop", ".3f"),
            # Harmonic
            KnobSpec("manifold_type",           "Manifold",   "choice", "pure", 0, 2, 1, "", _MANIFOLD, False, "Harmonic", ".0f", "", True),
            KnobSpec("harmonic_count",          "Count",      "int",   8,   1,   32,  1, "",   [], False, "Harmonic", ".0f", "HarmonicMeasureManifold"),
            KnobSpec("harmonic_brightness",     "Brightness", "float", 1.0, 0.0, 4.0, 0, "",   [], False, "Harmonic", ".2f"),
            KnobSpec("harmonic_warp_strength",  "Warp str.",  "float", 0.0, 0.0, 2.0, 0, "",   [], False, "Harmonic", ".3f", "PhaseWarpedManifold"),
            # Emission mode
            KnobSpec("emission_mode", "Emission", "choice", "single", 0, 1, 1, "",
                     ["single", "granular"], False, "Emission", ".0f", "", True),
            # Granular population controls (visible only when emission_mode == "granular")
            KnobSpec("granular.center_frequency_hz",         "Center freq",   "float", 440.0,  20.0,   20000.0, 0, "Hz", [], True,  "Granular: Freq",     ".1f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_pitch_spread_semitones","Pitch spread",  "float", 3.0,    0.0,    24.0,    0, "st", [], False, "Granular: Freq",     ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_harmonic_lock",         "Harm. lock",    "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Freq",     ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_highband_bias",         "Highband bias", "float", 0.0,    0.0,    3.0,     0, "oct",[], False, "Granular: Freq",     ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_density_hz",            "Density",       "float", 20.0,   0.5,    500.0,   0, "/s", [], True,  "Granular: Birth",    ".1f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.birth_jitter",                "Jitter",        "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Granular: Birth",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_probability",           "Burst prob.",   "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Birth",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_size",                  "Burst size",    "int",   4,      2,      32,      1, "",   [], False, "Granular: Birth",    ".0f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.burst_spread_s",              "Burst spread",  "float", 0.015,  0.001,  0.2,     0, "s",  [], True,  "Granular: Birth",    ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_duration_s",            "Duration",      "float", 0.05,   0.002,  2.0,     0, "s",  [], True,  "Granular: Duration", ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_duration_jitter",       "Dur. jitter",   "float", 0.3,    0.0,    2.0,     0, "",   [], False, "Granular: Duration", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_chirp_depth",           "Chirp depth",   "float", 0.0,    0.0,    2.0,     0, "",   [], False, "Granular: Chirp",    ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_chirp_jitter",          "Chirp jitter",  "float", 0.5,    0.0,    1.0,     0, "",   [], False, "Granular: Chirp",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_phase_randomness",      "Phase rand.",   "float", 1.0,    0.0,    1.0,     0, "",   [], False, "Granular: Phase",    ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_manifold_mix",          "Manifold mix",  "float", 0.0,    0.0,    1.0,     0, "",   [], False, "Granular: Manifold", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.gain",                        "Grain gain",    "float", 1.0,    0.0,    4.0,     0, "",   [], False, "Granular: Amp",      ".3f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_amp_jitter",            "Amp jitter",    "float", 0.2,    0.0,    2.0,     0, "",   [], False, "Granular: Amp",      ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.attack_frac",                 "Atk frac.",     "float", 0.15,   0.01,   0.5,     0, "",   [], False, "Granular: Envelope", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.release_frac",                "Rel frac.",     "float", 0.35,   0.01,   0.8,     0, "",   [], False, "Granular: Envelope", ".2f",  "", False, ("emission_mode", "granular")),
            KnobSpec("granular.grain_coherence",   "Coherence",    "float",  0.5, 0.0,  1.0,  0, "",   [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_mode",    "Coh. mode",    "choice", "uniform", 0, 0, 0, "",
                     ["uniform", "sine", "random_walk", "burst", "gradient", "perlin_walk"],
                     False, "Granular: Meta", "", "", True, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_rate_hz", "Coh. rate",    "float",  0.5, 0.01, 10.0, 0, "Hz", [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
            KnobSpec("granular.coherence_depth",   "Coh. depth",   "float",  0.3, 0.0,  1.0,  0, "",   [], False, "Granular: Meta", ".2f", "", False, ("emission_mode", "granular")),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticVoice":
        p = cls.__new__(cls)
        p.key          = d.get("key", uuid.uuid4().hex[:8])
        p.label        = d.get("label", "Voice")
        p.freq_hz      = float(d.get("freq_hz", 440.0))
        p.semitone_offset = float(d.get("semitone_offset", 0.0))
        p.note_tracking   = d.get("note_tracking", "note")
        if p.note_tracking not in ("note", "root", "free"):
            p.note_tracking = "note"
        p.amplitude    = float(d.get("amplitude", 1.0))
        p.phase_origin = float(d.get("phase_origin", 0.0))
        cd = d.get("chirp", {})
        p.chirp = ChirpSpec(
            f_delta_start=float(cd.get("f_delta_start", 0.0)),
            f_delta_end=float(cd.get("f_delta_end", 0.0)),
            chirp_type=cd.get("chirp_type", "none"),
            tau=float(cd.get("tau", 0.5)),
            chirp_power=float(cd.get("chirp_power", 1.0)),
        )
        fd = d.get("fm")
        p.fm = ModRouting(**fd) if fd else None
        ad = d.get("am")
        p.am = ModRouting(**ad) if ad else None
        p.env_type    = "piecewise"
        ad2 = d.get("adsr", {})
        p.adsr = ADSRParams(
            attack=float(ad2.get("attack",  0.005)),
            decay=float(ad2.get("decay",   0.04)),
            sustain=float(ad2.get("sustain", 0.75)),
            release=float(ad2.get("release", 0.08)),
            peak=float(ad2.get("peak", 1.0)),
        )
        p.env_knots    = d.get("env_knots", [[0,0],[0.01,1],[0.1,.75],[.85,.75],[1,0]])
        p.piecewise_env = PiecewiseVoiceEnvelope.from_dict(d.get("piecewise_env"))
        p.loop_start   = float(d.get("loop_start", 0.1))
        p.loop_end     = float(d.get("loop_end",   0.9))
        p.loop_enabled = bool(d.get("loop_enabled", False))
        p.muted        = bool(d.get("muted", False))
        p.color        = d.get("color", [100, 160, 255])
        p.pre_delay           = float(d.get("pre_delay", 0.0))
        p.manifold_type       = d.get("manifold_type", "pure")
        p.harmonic_count      = int(d.get("harmonic_count", 8))
        p.harmonic_brightness = float(d.get("harmonic_brightness", 1.0))
        p.harmonic_warp_strength = float(d.get("harmonic_warp_strength", 0.0))
        p.voice_role          = d.get("voice_role", "signal")
        p.seq_role            = d.get("seq_role", "melody")
        if p.seq_role not in ("melody", "bass", "root", "stab"):
            p.seq_role = "melody"
        p.register            = d.get("register", "all")
        if p.register not in ("all", "bass", "mid", "high"):
            p.register = "all"
        p.polyphony_count     = max(1, min(16, int(d.get("polyphony_count", 1))))
        p.polyphony_mode      = str(d.get("polyphony_mode", "sympathetic"))
        if p.polyphony_mode not in ("sympathetic", "unsympathetic"):
            p.polyphony_mode = "sympathetic"
        p.body_type           = str(d.get("body_type", "direct"))
        if p.body_type not in ("direct", "string_plate", "reed_box", "brass_bell", "drum_shell", "pipe_column", "voice_body"):
            p.body_type = "direct"
        p.emission_mode       = d.get("emission_mode", "single")
        # L2 fix: validate enum-like string fields — silently fall back to the
        # default rather than loading a typo that would synthesize silence.
        _VALID_EMISSION_MODES  = {"single", "granular"}
        _VALID_MANIFOLD_TYPES  = {"pure", "harmonic", "harmonic_warp"}
        if p.emission_mode not in _VALID_EMISSION_MODES:
            import warnings
            warnings.warn(
                f"AnalyticVoice.from_dict: unknown emission_mode {p.emission_mode!r}; "
                f"defaulting to 'single'.", stacklevel=2)
            p.emission_mode = "single"
        if p.manifold_type not in _VALID_MANIFOLD_TYPES:
            import warnings
            warnings.warn(
                f"AnalyticVoice.from_dict: unknown manifold_type {p.manifold_type!r}; "
                f"defaulting to 'pure'.", stacklevel=2)
            p.manifold_type = "pure"
        if p.piecewise_env is None:
            p.piecewise_env = PiecewiseVoiceEnvelope()
        raw_gran              = d.get("granular")
        if raw_gran and _HAS_GRANULAR:
            try:
                p.granular = _GrainPopulationSpec.from_dict(raw_gran)
            except Exception:
                p.granular = None
        else:
            p.granular = None
        return p


@dataclass
class PerformerPlacement:
    """One humanized performer instance derived from a chair assignment."""
    key: str
    label: str
    chair_key: str
    chair_index: int = 1
    performer_index: int = 1
    source_voice_keys: list = field(default_factory=list)
    source_layer_keys: list = field(default_factory=list)
    assigned_note_keys: list = field(default_factory=list)
    body_type: str = "direct"
    x: float = 0.0
    y: float = 0.0
    z: float = 1.1   # height above stage floor (meters); 1.1 = seated player reference
    face_x: float = 0.0    # aperture normal X — instrument faces conductor at origin
    face_y: float = -1.0   # aperture normal Y
    face_z: float = 0.0    # aperture normal Z
    radius: float = 0.0
    angle_deg: float = 0.0
    geometric_delay_ms: float = 0.0
    humanization_ms: float = 0.0
    phase_offset_rad: float = 0.0
    gain_db: float = 0.0
    pan: float = 0.0


@dataclass
class Chair:
    """A chair section such as 1st chair / 2nd chair within one Part."""
    key: str
    label: str
    part_key: str
    chair_index: int = 1
    specificity_rank: int = 0
    source_voice_keys: list = field(default_factory=list)
    source_layer_keys: list = field(default_factory=list)
    performer_count: int = 1
    performers: list = field(default_factory=list)  # list[PerformerPlacement]
    solver_hints: dict = field(default_factory=dict)


@dataclass
class NoteTarget:
    """The most holistic entity to which a NoteEvent is dispatched.

    The dispatch chain (most → least coordinated):
      "performer" — PerformerPlacement: knows position, delay, phase, gain, pan.
                    Multiple performers create a spatially-spread ensemble sound.
      "chair"     — Chair section: instrument-level grouping without per-seat
                    placement (uses shared voice signal, no geometric transforms).
      "voice"     — AnalyticVoice direct: no spatial context; raw synthesis only.

    ``voices`` is always the resolved list of AnalyticVoice objects that will
    actually synthesize audio regardless of the target_type chosen.
    """
    target_type: str              # "performer" | "chair" | "voice"
    voices: list                  # list[AnalyticVoice]
    performers: list              # list[PerformerPlacement]  — non-empty iff target_type=="performer"
    chairs: list                  # list[Chair]               — non-empty iff target_type in ("performer","chair")
    part: object                  # Part | None


@dataclass
class Part:
    """A resolved orchestral part, derived from the arrangement solver."""
    key: str                          # unique id, e.g. "bass-signal" or "high-melody"
    label: str                        # display name
    register: str                     # "bass" | "mid" | "high" | "all"
    seq_role: str                     # "melody" | "bass" | "root" | "stab" | ""
    voice_role: str                   # "signal" | "air" | "transient" | "body" | ""
    voice_keys: list = field(default_factory=list)   # AnalyticVoice.key values
    player_count: int = 1             # default one performer per part
    # Solver-derived performance envelope hint (optional, can be empty dict)
    solver_hints: dict = field(default_factory=dict)
    chairs: list = field(default_factory=list)       # list[Chair]


@dataclass
class PlacementResonatorConfig:
    """Patch-level placement-owned room/resonator defaults for deployed physics."""

    enabled: bool = False
    room_shape: str = "polygon"
    scene_path: str = ""
    room_radius: float = 3.4
    room_height: float = 3.6
    feedback_iterations: int = 1
    feedback_gain: float = 0.16
    passive_loss: float = 0.48
    band_split_mode: str = "fir"
    fir_taps: int = 65
    high_cone_deg: float = 70.0
    diffuse_strength: float = 0.42
    air_db_per_m: float = 0.01
    air_highband_db_per_m: float = 0.02
    temperature_c: float = 20.0
    humidity_rel: float = 0.5
    deployed_module_key: str = ""
    owner_module_type: str = "placement"
    # Mic array preset key from mic_arrays registry.
    # Empty string = legacy stereo pair (binaural_standard is the default when non-empty).
    receiver_array_key: str = "binaural_standard"
    # World-space receiver array center (meters, same coord space as room).
    receiver_pos_x: float = 0.0
    receiver_pos_y: float = 0.0
    receiver_pos_z: float = 1.5
    # Forward direction the array faces (normalized at use time).
    receiver_fwd_x: float = 0.0
    receiver_fwd_y: float = 1.0
    receiver_fwd_z: float = 0.0
    # Layout mode for performer packing.
    # "auto" = register-based semicircle (existing behaviour).
    # "stage" = dome-backed concert stage with proper orchestral section rows:
    #   1st/2nd Violins front-left arc, Violas center, Cellos right, Basses rear-right,
    #   Woodwinds center-rear, Brass right-rear, Percussion left-rear.
    layout_mode: str = "auto"

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "room_shape": str(self.room_shape),
            "scene_path": str(self.scene_path),
            "room_radius": float(self.room_radius),
            "room_height": float(self.room_height),
            "feedback_iterations": int(self.feedback_iterations),
            "feedback_gain": float(self.feedback_gain),
            "passive_loss": float(self.passive_loss),
            "band_split_mode": str(self.band_split_mode),
            "fir_taps": int(self.fir_taps),
            "high_cone_deg": float(self.high_cone_deg),
            "diffuse_strength": float(self.diffuse_strength),
            "air_db_per_m": float(self.air_db_per_m),
            "air_highband_db_per_m": float(self.air_highband_db_per_m),
            "temperature_c": float(self.temperature_c),
            "humidity_rel": float(self.humidity_rel),
            "deployed_module_key": str(self.deployed_module_key),
            "owner_module_type": str(self.owner_module_type),
            "receiver_array_key": str(self.receiver_array_key),
            "receiver_pos_x": float(self.receiver_pos_x),
            "receiver_pos_y": float(self.receiver_pos_y),
            "receiver_pos_z": float(self.receiver_pos_z),
            "receiver_fwd_x": float(self.receiver_fwd_x),
            "receiver_fwd_y": float(self.receiver_fwd_y),
            "receiver_fwd_z": float(self.receiver_fwd_z),
            "layout_mode": str(self.layout_mode),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementResonatorConfig":
        cfg = cls()
        cfg.enabled = bool(d.get("enabled", True))
        cfg.room_shape = str(d.get("room_shape", "polygon"))
        if cfg.room_shape not in ("polygon", "circular", "obj_mesh"):
            cfg.room_shape = "polygon"
        cfg.scene_path = str(d.get("scene_path", ""))
        cfg.room_radius = max(1.0, float(d.get("room_radius", 3.4)))
        cfg.room_height = max(1.5, float(d.get("room_height", 3.6)))
        cfg.feedback_iterations = max(0, int(d.get("feedback_iterations", 1)))
        cfg.feedback_gain = max(0.0, float(d.get("feedback_gain", 0.16)))
        cfg.passive_loss = max(0.0, min(0.98, float(d.get("passive_loss", 0.48))))
        cfg.band_split_mode = str(d.get("band_split_mode", "fir"))
        if cfg.band_split_mode not in ("fir", "fft"):
            cfg.band_split_mode = "fir"
        cfg.fir_taps = max(5, int(d.get("fir_taps", 65)) | 1)
        cfg.high_cone_deg = max(5.0, min(180.0, float(d.get("high_cone_deg", 70.0))))
        cfg.diffuse_strength = max(0.0, float(d.get("diffuse_strength", 0.42)))
        cfg.air_db_per_m = max(0.0, float(d.get("air_db_per_m", 0.01)))
        cfg.air_highband_db_per_m = max(0.0, float(d.get("air_highband_db_per_m", 0.02)))
        cfg.temperature_c = float(d.get("temperature_c", 20.0))
        cfg.humidity_rel = max(0.0, min(1.0, float(d.get("humidity_rel", 0.5))))
        cfg.deployed_module_key = str(d.get("deployed_module_key", ""))
        cfg.owner_module_type = str(d.get("owner_module_type", "placement") or "placement")
        cfg.receiver_array_key = str(d.get("receiver_array_key", "binaural_standard"))
        cfg.receiver_pos_x = float(d.get("receiver_pos_x", 0.0))
        cfg.receiver_pos_y = float(d.get("receiver_pos_y", 0.0))
        cfg.receiver_pos_z = float(d.get("receiver_pos_z", 1.5))
        cfg.receiver_fwd_x = float(d.get("receiver_fwd_x", 0.0))
        cfg.receiver_fwd_y = float(d.get("receiver_fwd_y", 1.0))
        cfg.receiver_fwd_z = float(d.get("receiver_fwd_z", 0.0))
        cfg.layout_mode = str(d.get("layout_mode", "auto"))
        if cfg.layout_mode not in ("auto", "stage"):
            cfg.layout_mode = "auto"
        return cfg


@dataclass
class LFODefinition:
    ARCHETYPE_KEY: ClassVar[str] = "analytic.lfo"

    key:            str       = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:          str       = "LFO"
    # Slot 0 params — stored at top level for UI / knob-system compat.
    rate_hz:        float     = 1.0
    shape:          str       = "Sine"   # Sine | Triangle | Sawtooth | Square
    phase_offset:   float     = 0.0
    depth:          float     = 1.0
    # Packing capacity: number of parallel LFO slots this node carries.
    # Slot 0 uses the top-level scalar fields above; slots 1..capacity-1 are
    # stored in extra_channels as dicts {rate_hz, shape, phase_offset, depth}.
    capacity:       int       = 1
    extra_channels: list      = field(default_factory=list)
    color: list[int] = field(default_factory=lambda: [200, 160, 60])

    _SLOT_DEFAULTS: ClassVar[dict] = {
        "rate_hz": 1.0, "shape": "Sine", "phase_offset": 0.0, "depth": 1.0,
    }

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"rate_hz", "depth", "shape", "phase_offset"}),
            tracked_inputs=frozenset(),
        )

    def _all_channels(self) -> list:
        """Return a list of per-slot param dicts of length ``capacity``.

        Slot 0 reflects the top-level scalar fields.  Slots 1..capacity-1
        come from ``extra_channels``, padded with defaults if needed.
        """
        slot0 = {"rate_hz": self.rate_hz, "shape": self.shape,
                 "phase_offset": self.phase_offset, "depth": self.depth}
        extras = list(self.extra_channels)
        while len(extras) < max(0, self.capacity - 1):
            extras.append(dict(self._SLOT_DEFAULTS))
        return [slot0] + extras[:max(0, self.capacity - 1)]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "rate_hz": self.rate_hz, "shape": self.shape,
                "phase_offset": self.phase_offset, "depth": self.depth,
                "capacity": self.capacity,
                "extra_channels": list(self.extra_channels),
                "color": self.color}

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _SHAPES = ["Sine", "Triangle", "Sawtooth", "Square"]
        return [
            KnobSpec("capacity",     "Capacity",     "int",    1,   1,   64,       1, "",    [], False, "LFO", ".0f",
                     True),
            KnobSpec("rate_hz",      "Rate (slot 0)","float",  1.0, 0.01, 50.0,   0, "Hz",  [], True,  "LFO", ".3f"),
            KnobSpec("phase_offset", "Phase (slot 0)","float", 0.0,-math.pi, math.pi, 0, "rad", [], False, "LFO", ".3f"),
            KnobSpec("depth",        "Depth (slot 0)","float", 1.0, 0.0, 2.0,     0, "",    [], False, "LFO", ".3f"),
            KnobSpec("shape",        "Shape (slot 0)","choice","Sine", 0, 3, 1, "", _SHAPES, False, "LFO"),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "LFODefinition":
        o = cls.__new__(cls)
        o.key            = d.get("key", uuid.uuid4().hex[:8])
        o.label          = d.get("label", "LFO")
        o.rate_hz        = float(d.get("rate_hz", 1.0))
        o.shape          = d.get("shape", "Sine")
        o.phase_offset   = float(d.get("phase_offset", 0.0))
        o.depth          = float(d.get("depth", 1.0))
        o.capacity       = int(d.get("capacity", 1))
        o.extra_channels = list(d.get("extra_channels", []))
        o.color          = d.get("color", [200, 160, 60])
        return o
# ---------------------------------------------------------------------------
# Temperament interval tables (module-level; shared by GlobalTuning).
# ---------------------------------------------------------------------------
_JUST_INTERVAL_RATIOS: list = [
    1.0, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 16/9, 15/8,
]
_PYTHAGOREAN_RATIOS: list = [
    1.0, 256/243, 9/8, 32/27, 81/64, 4/3, 729/512, 3/2, 128/81, 27/16, 16/9, 243/128,
]

# 22 śrutis of the Indian classical system (Bharata / Natya Shastra), in cents.
# Ordered chromatically; index = sruti number (0-based).
_22_SRUTI_CENTS: list = [
    0.000,      #  0 Sa            (tonic)
    90.225,     #  1 komal Re-1    (ek sruti)
    111.731,    #  2 komal Re-2    (do sruti)
    182.404,    #  3 Re-1          (tri sruti)
    203.910,    #  4 shuddha Re    (chatur sruti Rishab)
    294.135,    #  5 komal Ga-1    (sadharana Gandhar low)
    315.641,    #  6 komal Ga-2    (sadharana Gandhar / komal Ga)
    386.314,    #  7 antara Ga     (shuddha Ga in common usage)
    407.820,    #  8 shuddha Ga-2  (chatur sruti Gandhar)
    498.045,    #  9 shuddha Ma    (perfect fourth)
    519.551,    # 10 Ma-2
    590.224,    # 11 tivra Ma-1
    611.730,    # 12 tivra Ma-2    (tritone)
    701.955,    # 13 Pa            (perfect fifth)
    792.180,    # 14 komal Dha-1   (ek sruti)
    813.686,    # 15 komal Dha-2   (do sruti)
    884.359,    # 16 Dha-1         (tri sruti)
    905.865,    # 17 shuddha Dha   (chatur sruti Dhaivat)
    996.090,    # 18 komal Ni-1    (ek sruti)
    1017.596,   # 19 komal Ni-2    (kaisiki Nishad)
    1088.269,   # 20 Ni-1          (kakali Nishad low)
    1109.775,   # 21 shuddha Ni    (kakali Nishad / Ni-2)
]

# ¼-comma meantone cents for 12 pitch classes from the tonic.
_MEANTONE_QC_CENTS: list = [
    0.000, 76.049, 193.157, 269.205, 386.314, 503.422,
    579.471, 696.578, 772.627, 889.735, 965.784, 1082.892,
]

# Kirnberger III well-temperament cents from C.
_KIRNBERGER_III_CENTS: list = [
    0.000, 90.225, 193.157, 294.135, 386.314, 498.045,
    590.224, 696.578, 792.180, 889.735, 996.090, 1082.892,
]

# ---------------------------------------------------------------------------
# Global tuning presets.
# Each entry is a plain dict with all GlobalTuning field values plus a human-
# readable "label".  Keys are stable identifiers used as preset names.
# ---------------------------------------------------------------------------
GLOBAL_TUNING_PRESETS: dict = {
    # ── Standard Western ────────────────────────────────────────────────────
    "a440_12tet": {
        "label":         "A=440  12-TET (standard)",
        "root_hz":       440.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a432_12tet": {
        "label":         "A=432  12-TET (Verdi / alternative concert pitch)",
        "root_hz":       432.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a415_12tet": {
        "label":         "A=415  12-TET (Baroque low pitch)",
        "root_hz":       415.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Historical temperaments ──────────────────────────────────────────────
    "just_a440": {
        "label":         "A=440  Just Intonation (5-limit)",
        "root_hz":       440.0,
        "temperament":   "just",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "pythagorean_a440": {
        "label":         "A=440  Pythagorean (3-limit)",
        "root_hz":       440.0,
        "temperament":   "pythagorean",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "meantone_a440": {
        "label":         "A=440  ¼-comma Meantone",
        "root_hz":       440.0,
        "temperament":   "custom",
        "custom_cents":  list(_MEANTONE_QC_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "well_tempered_c": {
        "label":         "C=261.63  Kirnberger III Well-Temperament",
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_KIRNBERGER_III_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Indian classical (22-śruti) ──────────────────────────────────────────
    "raga_bhairav": {
        "label":         "Sa=261.63  Raga Bhairav (22 śrutis)",
        # Bhairav: Sa komal-Re Ga Ma Pa komal-Dha Ni
        # Evokes dawn; gravity, devotion, restraint.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 7, 9, 13, 14, 20],
        "scale_name":    "raga_bhairav",
    },
    "raga_yaman": {
        "label":         "Sa=261.63  Raga Yaman / Kalyan (22 śrutis)",
        # Yaman: Sa Re Ga tivra-Ma Pa Dha Ni  (Lydian-like)
        # Evening raga; floating, expansive, contemplative.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 4, 7, 11, 13, 17, 20],
        "scale_name":    "raga_yaman",
    },
    "raga_bhairavi": {
        "label":         "Sa=261.63  Raga Bhairavi (22 śrutis)",
        # Bhairavi: Sa komal-Re komal-Ga Ma Pa komal-Dha komal-Ni
        # (All-flat Phrygian-like; morning farewell, melancholic beauty.)
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 6, 9, 13, 14, 19],
        "scale_name":    "raga_bhairavi",
    },
}

# 22 śrutis of the Indian classical system (Bharata / Natya Shastra), in cents.
# Ordered chromatically; index = sruti number (0-based).
_22_SRUTI_CENTS: list = [
    0.000,      #  0 Sa            (tonic)
    90.225,     #  1 komal Re-1    (ek sruti)
    111.731,    #  2 komal Re-2    (do sruti)
    182.404,    #  3 Re-1          (tri sruti)
    203.910,    #  4 shuddha Re    (chatur sruti Rishab)
    294.135,    #  5 komal Ga-1    (sadharana Gandhar low)
    315.641,    #  6 komal Ga-2    (sadharana Gandhar / komal Ga)
    386.314,    #  7 antara Ga     (shuddha Ga in common usage)
    407.820,    #  8 shuddha Ga-2  (chatur sruti Gandhar)
    498.045,    #  9 shuddha Ma    (perfect fourth)
    519.551,    # 10 Ma-2
    590.224,    # 11 tivra Ma-1
    611.730,    # 12 tivra Ma-2    (tritone)
    701.955,    # 13 Pa            (perfect fifth)
    792.180,    # 14 komal Dha-1   (ek sruti)
    813.686,    # 15 komal Dha-2   (do sruti)
    884.359,    # 16 Dha-1         (tri sruti)
    905.865,    # 17 shuddha Dha   (chatur sruti Dhaivat)
    996.090,    # 18 komal Ni-1    (ek sruti)
    1017.596,   # 19 komal Ni-2    (kaisiki Nishad)
    1088.269,   # 20 Ni-1          (kakali Nishad low)
    1109.775,   # 21 shuddha Ni    (kakali Nishad / Ni-2)
]

# ¼-comma meantone cents for 12 pitch classes from the tonic.
_MEANTONE_QC_CENTS: list = [
    0.000, 76.049, 193.157, 269.205, 386.314, 503.422,
    579.471, 696.578, 772.627, 889.735, 965.784, 1082.892,
]

# Kirnberger III well-temperament cents from C.
_KIRNBERGER_III_CENTS: list = [
    0.000, 90.225, 193.157, 294.135, 386.314, 498.045,
    590.224, 696.578, 792.180, 889.735, 996.090, 1082.892,
]

# ---------------------------------------------------------------------------
# Global tuning presets.
# Each entry is a plain dict with all GlobalTuning field values plus a human-
# readable "label".  Keys are stable identifiers used as preset names.
# ---------------------------------------------------------------------------
GLOBAL_TUNING_PRESETS: dict = {
    # ── Standard Western ────────────────────────────────────────────────────
    "a440_12tet": {
        "label":         "A=440  12-TET (standard)",
        "root_hz":       440.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a432_12tet": {
        "label":         "A=432  12-TET (Verdi / alternative concert pitch)",
        "root_hz":       432.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "a415_12tet": {
        "label":         "A=415  12-TET (Baroque low pitch)",
        "root_hz":       415.0,
        "temperament":   "12tet",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Historical temperaments ──────────────────────────────────────────────
    "just_a440": {
        "label":         "A=440  Just Intonation (5-limit)",
        "root_hz":       440.0,
        "temperament":   "just",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "pythagorean_a440": {
        "label":         "A=440  Pythagorean (3-limit)",
        "root_hz":       440.0,
        "temperament":   "pythagorean",
        "custom_cents":  [0,100,200,300,400,500,600,700,800,900,1000,1100],
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "meantone_a440": {
        "label":         "A=440  ¼-comma Meantone",
        "root_hz":       440.0,
        "temperament":   "custom",
        "custom_cents":  list(_MEANTONE_QC_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    "well_tempered_c": {
        "label":         "C=261.63  Kirnberger III Well-Temperament",
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_KIRNBERGER_III_CENTS),
        "scale_degrees": list(range(12)),
        "scale_name":    "chromatic",
    },
    # ── Indian classical (22-śruti) ──────────────────────────────────────────
    "raga_bhairav": {
        "label":         "Sa=261.63  Raga Bhairav (22 śrutis)",
        # Bhairav: Sa komal-Re Ga Ma Pa komal-Dha Ni
        # Evokes dawn; gravity, devotion, restraint.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 7, 9, 13, 14, 20],
        "scale_name":    "raga_bhairav",
    },
    "raga_yaman": {
        "label":         "Sa=261.63  Raga Yaman / Kalyan (22 śrutis)",
        # Yaman: Sa Re Ga tivra-Ma Pa Dha Ni  (Lydian-like)
        # Evening raga; floating, expansive, contemplative.
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 4, 7, 11, 13, 17, 20],
        "scale_name":    "raga_yaman",
    },
    "raga_bhairavi": {
        "label":         "Sa=261.63  Raga Bhairavi (22 śrutis)",
        # Bhairavi: Sa komal-Re komal-Ga Ma Pa komal-Dha komal-Ni
        # (All-flat Phrygian-like; morning farewell, melancholic beauty.)
        "root_hz":       261.626,
        "temperament":   "custom",
        "custom_cents":  list(_22_SRUTI_CENTS),
        "scale_degrees": [0, 1, 6, 9, 13, 14, 19],
        "scale_name":    "raga_bhairavi",
    },
}


# ---------------------------------------------------------------------------
# GlobalTuning — pitch reference frame for an AnalyticPatch.
#
# Defines what "semitone 0" means (root_hz), how semitone-to-Hz conversion is
# performed (temperament), and which pitch classes are active (scale_degrees /
# scale_name).  Voices and the PitchQuantizer both consult this object so that
# one global change (e.g. transposing root_hz) cascades everywhere.
# ---------------------------------------------------------------------------
@dataclass
class GlobalTuning:
    """Pitch reference frame for an AnalyticPatch.

    ``divisions_per_octave`` is a *derived* property — it equals
    ``len(custom_cents)`` when ``temperament == "custom"`` and 12 otherwise.
    This means changing ``custom_cents`` to a 22-entry śruti table automatically
    makes every offset/quantize operation work in 22-per-octave space.

    When using a preset via :meth:`from_preset`, ``scale_name`` and all related
    fields are set together so nothing falls out of sync.
    """
    root_hz:       float = 440.0
    temperament:   str   = "12tet"   # "12tet" | "just" | "pythagorean" | "custom"
    custom_cents:  list  = field(default_factory=lambda: [
        0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100
    ])
    scale_degrees: list  = field(default_factory=lambda: list(range(12)))
    scale_name:    str   = "chromatic"
    preset_name:   str   = "a440_12tet"  # last-applied preset (display hint)

    _TEMPERAMENT_CHOICES = ["12tet", "just", "pythagorean", "custom"]

    # ── Derived property ────────────────────────────────────────────────────
    @property
    def divisions_per_octave(self) -> int:
        """Number of equal-or-custom steps that span one octave.

        For the three named temperaments this is always 12.
        For ``"custom"`` it equals ``len(custom_cents)``, allowing non-12
        systems (e.g. 22 Indian śrutis) simply by supplying a different
        ``custom_cents`` list.
        """
        if self.temperament == "custom":
            return max(1, len(self.custom_cents))
        return 12

    # ── Conversion methods ──────────────────────────────────────────────────
    def semitone_to_hz(self, semitones: float) -> float:
        """Convert *semitones* (steps relative to root_hz) to Hz.

        One "semitone" is ``1 / divisions_per_octave`` of an octave in the
        active temperament.  For 22-śruti custom tuning, ``semitones=22``
        is exactly one octave up.
        """
        dpo      = self.divisions_per_octave
        octaves  = math.floor(semitones / dpo)
        frac     = semitones - dpo * octaves
        degree   = int(round(frac)) % dpo
        leftover = frac - degree
        t = self.temperament
        if t == "just":
            ratio = _JUST_INTERVAL_RATIOS[degree % 12]
        elif t == "pythagorean":
            ratio = _PYTHAGOREAN_RATIOS[degree % 12]
        elif t == "custom" and self.custom_cents:
            idx   = degree % len(self.custom_cents)
            ratio = 2.0 ** (self.custom_cents[idx] / 1200.0)
        else:  # 12tet
            ratio = 2.0 ** (degree / 12.0)
        if abs(leftover) > 1e-9:
            ratio *= 2.0 ** (leftover / dpo)
        return self.root_hz * (2.0 ** octaves) * ratio

    def hz_to_semitones(self, hz: float) -> float:
        """Convert *hz* to steps (semitones) relative to root_hz.

        The result is in the same unit system as :meth:`semitone_to_hz`:
        one step = one division of the octave in the active tuning.
        """
        if hz <= 0.0:
            return 0.0
        return self.divisions_per_octave * math.log2(hz / self.root_hz)

    def quantize(self, semitones: float) -> float:
        """Snap *semitones* to nearest active scale degree (octave-preserving)."""
        dpo     = self.divisions_per_octave
        degrees = sorted(self.scale_degrees)
        if not degrees:
            return semitones
        octave  = math.floor(semitones / dpo)
        pc      = semitones - dpo * octave
        nearest = min(degrees, key=lambda d: abs(d - pc))
        if abs(degrees[0] + dpo - pc) < abs(nearest - pc):
            nearest = degrees[0]
            octave += 1
        return dpo * octave + float(nearest)

    # ── Preset system ───────────────────────────────────────────────────────
    @classmethod
    def from_preset(cls, name: str) -> "GlobalTuning":
        """Return a :class:`GlobalTuning` configured from a named preset.

        Available preset names are the keys of :data:`GLOBAL_TUNING_PRESETS`.
        Raises ``ValueError`` for unknown names.
        """
        data = GLOBAL_TUNING_PRESETS.get(name)
        if data is None:
            raise ValueError(
                f"Unknown GlobalTuning preset {name!r}.  "
                f"Available: {sorted(GLOBAL_TUNING_PRESETS)}"
            )
        o = cls.__new__(cls)
        o.root_hz       = float(data.get("root_hz",  440.0))
        o.temperament   = data.get("temperament",    "12tet")
        o.custom_cents  = list(data.get("custom_cents",
                                        [0,100,200,300,400,500,600,700,800,900,1000,1100]))
        o.scale_degrees = list(data.get("scale_degrees", list(range(12))))
        o.scale_name    = data.get("scale_name",     "chromatic")
        o.preset_name   = name
        return o

    # ── Serialisation ───────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "root_hz":       self.root_hz,
            "temperament":   self.temperament,
            "custom_cents":  list(self.custom_cents),
            "scale_degrees": list(self.scale_degrees),
            "scale_name":    self.scale_name,
            "preset_name":   self.preset_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalTuning":
        o = cls.__new__(cls)
        o.root_hz       = float(d.get("root_hz", 440.0))
        o.temperament   = d.get("temperament", "12tet")
        if o.temperament not in cls._TEMPERAMENT_CHOICES:
            o.temperament = "12tet"
        o.custom_cents  = list(d.get("custom_cents",
                                     [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]))
        o.scale_degrees = list(d.get("scale_degrees", list(range(12))))
        o.scale_name    = d.get("scale_name", "chromatic")
        o.preset_name   = d.get("preset_name", "")
        return o

    @classmethod
    def knobs(cls) -> list:
        _preset_choices = list(GLOBAL_TUNING_PRESETS.keys())
        return [
            KnobSpec("preset_name",  "Preset",      "choice", "a440_12tet", 0,
                     max(0, len(_preset_choices) - 1), 1, "",
                     _preset_choices, False, "Tuning", "", "", True),
            KnobSpec("root_hz",      "Root Hz",     "float",  440.0, 20.0, 8000.0, 0, "Hz", [],
                     True,  "Tuning", ".2f"),
            KnobSpec("temperament",  "Temperament", "choice", "12tet", 0, 3, 1, "",
                     cls._TEMPERAMENT_CHOICES, False, "Tuning", "", "", True),
            KnobSpec("scale_name",   "Scale",       "str",    "chromatic", 0, 0, 0, "",
                     [], False, "Tuning"),
        ]


# ---------------------------------------------------------------------------
# QuantizerHandle — callable pitch quantizer with pluggable interpolation.
#
# Returned by make_quantizer_handle().  The caller injects:
#   value          : float — raw input to quantize
#   original_value : float — pre-quantization source value (same domain);
#                            used by interpolators to track continuity across
#                            unquantized motion
#   domain         : str   — "semitone" (relative to tuning root) | "hz"
#   dt             : float — elapsed seconds since last call (time-based modes)
#
# All configuration is captured at construction time; _state is mutable.
# Call handle.reset() before replaying a note or resetting the patch.
# ---------------------------------------------------------------------------
class QuantizerHandle:
    """
    Callable pitch quantizer encapsulating scale, interpolation mode, and
    mutable integrator state.

    Call signature::

        result = handle(value, original_value, domain="semitone", dt=0.0)

    Parameters
    ----------
    value          : input to quantize (semitones or Hz per *domain*)
    original_value : pre-quantization source in the same *domain*
    domain         : ``"semitone"`` | ``"hz"``
    dt             : elapsed seconds since previous call (required for
                     ``portamento``, ``slew``, ``slew2``, ``spline``,
                     ``legato``; use ``0.0`` for stateless / one-shot use)

    Returns the quantized output in the **same domain** as *value*.
    """

    _INTERP_MODES = ["discrete", "portamento", "slew", "slew2", "spline", "legato"]
    ARCHETYPE_KEY: ClassVar[str] = "analytic.quantizer"

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"portamento_time", "slew_rate", "slew2_accel",
                                       "spline_tension", "scale_degrees"}),
            tracked_inputs=frozenset({"pitch_in"}),
        )

    def __init__(
        self,
        tuning:          "GlobalTuning",
        scale_degrees:   list,
        mode:            str   = "discrete",
        portamento_time: float = 0.05,
        slew_rate:       float = 100.0,
        slew2_accel:     float = 200.0,
        spline_tension:  float = 0.5,
    ) -> None:
        self._tuning          = tuning
        self._scale_degrees   = sorted(set(scale_degrees)) if scale_degrees else list(range(12))
        self._mode            = mode if mode in self._INTERP_MODES else "discrete"
        self._portamento_time = float(portamento_time)
        self._slew_rate       = float(slew_rate)
        self._slew2_accel     = float(slew2_accel)
        self._spline_tension  = float(spline_tension)
        self._state: dict = {
            "position":    None,   # current smoothed semitone output; None = uninitialised
            "velocity":    0.0,    # semitones/s  (portamento / slew / slew2)
            "target":      None,   # last quantized target semitone
            "target_prev": None,   # for change detection (spline / legato)
            "spline_pos0": 0.0,    # start position for current spline segment
            "spline_t":    0.0,    # normalised time 0→1 along current spline segment
        }

    @property
    def state(self) -> dict:
        return self._state

    # ── internal helpers ──────────────────────────────────────────────────
    def _quantize_st(self, semitones: float) -> float:
        dpo     = self._tuning.divisions_per_octave
        degrees = self._scale_degrees
        octave  = math.floor(semitones / dpo)
        pc      = semitones - dpo * octave
        nearest = min(degrees, key=lambda d: abs(d - pc))
        if abs(degrees[0] + dpo - pc) < abs(nearest - pc):
            nearest = degrees[0]
            octave += 1
        return dpo * octave + float(nearest)

    def _quantize_st_array(self, semitones: np.ndarray) -> np.ndarray:
        arr = np.asarray(semitones, dtype=np.float64)
        dpo = float(self._tuning.divisions_per_octave)
        degrees = np.asarray(self._scale_degrees, dtype=np.float64)
        if arr.size == 0:
            return arr.copy()
        octave = np.floor(arr / dpo)
        pc = arr - dpo * octave
        deltas = np.abs(pc[:, None] - degrees[None, :])
        nearest_idx = np.argmin(deltas, axis=1)
        nearest = degrees[nearest_idx]
        wrap_delta = np.abs((degrees[0] + dpo) - pc)
        wrap_mask = wrap_delta < np.abs(nearest - pc)
        nearest = nearest.copy()
        nearest[wrap_mask] = degrees[0]
        octave = octave.copy()
        octave[wrap_mask] += 1.0
        return dpo * octave + nearest

    def _to_st(self, value: float, domain: str) -> float:
        if domain == "hz":
            if value <= 0.0:
                return 0.0
            return 12.0 * math.log2(value / self._tuning.root_hz)
        return float(value)

    def _to_st_array(self, values: np.ndarray, domain: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if domain != "hz":
            return arr.copy()
        out = np.zeros_like(arr)
        pos = arr > 0.0
        if np.any(pos):
            out[pos] = 12.0 * np.log2(arr[pos] / self._tuning.root_hz)
        return out

    def _from_st(self, st: float, domain: str) -> float:
        if domain == "hz":
            return self._tuning.semitone_to_hz(st)
        return st

    def _from_st_array(self, st: np.ndarray, domain: str) -> np.ndarray:
        arr = np.asarray(st, dtype=np.float64)
        if domain != "hz":
            return arr.copy()
        return self._tuning.root_hz * np.power(2.0, arr / 12.0)

    # ── main entry point ──────────────────────────────────────────────────
    def __call__(
        self,
        value:          float,
        original_value: float,
        domain:         str   = "semitone",
        dt:             float = 0.0,
    ) -> float:
        """
        Quantize *value* against the active scale and apply interpolation.

        Parameters
        ----------
        value          : raw input in *domain*
        original_value : unquantized source in *domain* (used for interpolation
                         continuity; may equal *value* when the caller has no
                         separate pre-quantization signal)
        domain         : ``"semitone"`` | ``"hz"``
        dt             : seconds since last call; ``0.0`` → stateless snap
        """
        value_st    = self._to_st(value, domain)
        original_st = self._to_st(original_value, domain)
        target_st   = self._quantize_st(value_st)
        self._state["target"] = target_st

        pos = self._state["position"]
        if pos is None:
            pos = self._quantize_st(original_st)

        mode = self._mode

        if mode == "discrete" or dt <= 0.0:
            pos = target_st

        elif mode == "portamento":
            if self._portamento_time > 0.0:
                alpha = 1.0 - math.exp(-dt / self._portamento_time)
                pos   = pos + alpha * (target_st - pos)
            else:
                pos = target_st

        elif mode == "slew":
            max_delta = self._slew_rate * dt
            dist      = target_st - pos
            pos       = pos + math.copysign(min(abs(dist), max_delta), dist)

        elif mode == "slew2":
            vel  = self._state["velocity"]
            dist = target_st - pos
            sign = math.copysign(1.0, dist) if dist != 0.0 else 0.0
            accel = sign * self._slew2_accel - vel * 2.0
            vel  += accel * dt
            if abs(vel) * dt > abs(dist) and dist * vel > 0:
                vel = dist / max(dt, 1e-9)
            pos += vel * dt
            self._state["velocity"] = vel

        elif mode == "spline":
            if target_st != self._state["target_prev"]:
                self._state["spline_pos0"] = pos
                self._state["spline_t"]    = 0.0
                self._state["target_prev"] = target_st
            t = self._state["spline_t"]
            if self._portamento_time > 0.0 and dt > 0.0:
                t = min(t + dt / self._portamento_time, 1.0)
            else:
                t = 1.0
            tension = self._spline_tension
            p0 = self._state["spline_pos0"]
            m0 = (target_st - p0) * tension
            m1 = m0
            h00 =  2*t**3 - 3*t**2 + 1
            h10 =    t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 =    t**3 -   t**2
            pos = h00 * p0 + h10 * m0 + h01 * target_st + h11 * m1
            self._state["spline_t"] = t

        elif mode == "legato":
            if target_st != self._state["target_prev"]:
                self._state["target_prev"] = target_st
                if self._portamento_time > 0.0 and dt > 0.0:
                    alpha = 1.0 - math.exp(-dt / self._portamento_time)
                    pos   = pos + alpha * (target_st - pos)
                else:
                    pos = target_st
            # else: hold current position

        self._state["position"] = pos
        return self._from_st(pos, domain)

    def process_series(
        self,
        values: np.ndarray,
        original_values: "np.ndarray | None" = None,
        domain: str = "semitone",
        dt: float = 0.0,
    ) -> np.ndarray:
        """Quantize a full series, using vectorized math when possible."""
        vals = np.asarray(values, dtype=np.float64)
        orig = vals if original_values is None else np.asarray(original_values, dtype=np.float64)
        if vals.shape != orig.shape:
            raise ValueError("values and original_values must have the same shape")
        if vals.ndim != 1:
            raise ValueError("process_series expects a 1-D array")
        if vals.size == 0:
            return vals.copy()
        if self._mode == "discrete" or dt <= 0.0:
            value_st = self._to_st_array(vals, domain)
            target_st = self._quantize_st_array(value_st)
            self._state["target"] = float(target_st[-1])
            self._state["position"] = float(target_st[-1])
            return self._from_st_array(target_st, domain)
        out = np.empty_like(vals, dtype=np.float64)
        for i in range(vals.size):
            out[i] = self(vals[i], orig[i], domain=domain, dt=dt)
        return out

    def reset(self) -> None:
        """Clear all integrator state (call before a new note or patch reset)."""
        self._state["position"]    = None
        self._state["velocity"]    = 0.0
        self._state["target"]      = None
        self._state["target_prev"] = None
        self._state["spline_pos0"] = 0.0
        self._state["spline_t"]    = 0.0


def make_quantizer_handle(
    module:  "AnalyticModule",
    tuning:  "GlobalTuning",
) -> QuantizerHandle:
    """
    Build a :class:`QuantizerHandle` from a ``pitch_quantizer``
    :class:`AnalyticModule` and a :class:`GlobalTuning`.

    ``module.quantizer_scale_degrees`` provides the active pitch classes;
    when empty the tuning's own ``scale_degrees`` are used.
    """
    degrees = (sorted(set(module.quantizer_scale_degrees))
               if module.quantizer_scale_degrees
               else list(tuning.scale_degrees))
    return QuantizerHandle(
        tuning          = tuning,
        scale_degrees   = degrees,
        mode            = module.interpolation_mode,
        portamento_time = module.portamento_time,
        slew_rate       = module.slew_rate,
        slew2_accel     = module.slew2_accel,
        spline_tension  = module.spline_tension,
    )


# ---------------------------------------------------------------------------
# AnalyticMixer — a named mixer node in the routing graph.
#
# projection_active = True  → its output is summed into the PCM bus (speaker /
#                             file).  The patch-level projection_mode / rotation
#                             settings are applied before writing to the bus.
# projection_active = False → analytic meta-mixer only.  Its output is a
#                             complex analytic signal that can be routed into
#                             other mixer nodes but does NOT contribute to the
#                             PCM bus.  Useful for sub-mixes, sidechains, etc.
# ---------------------------------------------------------------------------
@dataclass
class AnalyticMixer:
    ARCHETYPE_KEY: ClassVar[str] = "analytic.mixer"

    key:               str  = field(default_factory=lambda: "__mix__")
    label:             str  = "Mix"
    projection_active: bool = True     # True → output track; False → meta-mixer
    color: list = field(default_factory=lambda: [200, 200, 100, 255])
    # File export settings (used by the Render button in the sequencer panel)
    export_to_file:     bool = False
    export_sample_rate: int  = 48000
    export_bit_depth:   int  = 24
    # Signal layer this mixer operates at.  The only valid mixer layer is
    # "master" — the single top-level mix bus that receives room SM mic
    # streams and any instrument signals routed directly here.
    mixer_layer: str = "master"

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"projection_active", "mixer_layer"}),
            tracked_inputs=frozenset({"signal_in"}),
        )

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "projection_active": self.projection_active,
                "color": self.color,
                "export_to_file":     self.export_to_file,
                "export_sample_rate": self.export_sample_rate,
                "export_bit_depth":   self.export_bit_depth,
                "mixer_layer":        self.mixer_layer}

    @classmethod
    def knobs(cls) -> list["KnobSpec"]:
        return [
            KnobSpec("label",             "Label",      "str",  "Mix", 0, 0, 0, "", [], False, "Mixer"),
            KnobSpec("projection_active", "Output/PCM", "bool", True,  0, 1, 0, "", [], False, "Mixer"),
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticMixer":
        o = cls.__new__(cls)
        o.key               = d.get("key", "__mix__")
        o.label             = d.get("label", "Mix")
        o.projection_active = bool(d.get("projection_active", True))
        o.color             = d.get("color", [200, 200, 100, 255])
        o.export_to_file     = bool(d.get("export_to_file",     False))
        o.export_sample_rate = int(d.get("export_sample_rate",  48000))
        o.export_bit_depth   = int(d.get("export_bit_depth",    24))
        o.mixer_layer        = str(d.get("mixer_layer", "master"))
        return o


@dataclass
class PortTensorSpec:
    """Negotiation metadata for one published port.

    tensor_rank
        Conceptual rank of the payload.  0 = scalar control, 1 = vector lane,
        2+ = structured tensor batch.  This is descriptive for now and lets the
        graph/UI evolve toward massive parallel cables without changing the
        publishing API again.
    lane_count
        Number of parallel lanes exposed by this port when known.  0 means
        "dynamic / negotiated at compile time".
    parallel_group
        Optional symbolic grouping key used to validate wide-bus compatibility
        across related ports (for example room↔performer state exchange).
    """
    tensor_rank: int = 0
    lane_count: int = 1
    batch_axes: int = 1
    parallel_group: str = ""
    group_validity: str = "strict"   # strict | broadcast | reduce | remap
    semantic_role: str = ""
    channel_dims: list = field(default_factory=list)
    batchable: bool = True
    dtype: str = "complex128"
    analytic_only: bool = True

    def to_contract(self) -> "TensorPortContract":
        return TensorPortContract(
            dtype=str(self.dtype or "complex128"),
            analytic_only=bool(self.analytic_only),
            tensor_rank=max(0, int(self.tensor_rank)),
            lane_count=max(0, int(self.lane_count)),
            batch_axes=max(0, int(self.batch_axes)),
            parallel_group=str(self.parallel_group or ""),
            group_validity=str(self.group_validity or "strict"),
            semantic_role=str(self.semantic_role or ""),
            channel_dims=[int(x) for x in self.channel_dims],
        )


@dataclass
class PublishedPort:
    key: str = ""
    label: str = ""
    direction: str = "out"      # "in" | "out"
    domain: str = "control"     # "signal" | "control" | "param_target"
    owner_key: str = ""
    group: str = ""
    param_path: str = ""
    color: tuple[int, int, int] = (150, 150, 170)
    tensor: PortTensorSpec = field(default_factory=PortTensorSpec)
    semantic_role: str = ""
    projection_policy: str = ""   # only meaningful for scalar / non-complex destinations
    negotiates_group_validity: bool = False


@dataclass
class RackPortView:
    port: PublishedPort
    local_x: int = 0
    local_y: int = 0
    radius: int = 3


@dataclass
class RackDeviceView:
    device_key: str = ""
    label: str = ""
    device_kind: str = ""
    color: tuple[int, int, int] = (90, 110, 140)
    rack_u: int = 1
    rack_w: int = 1
    grid_x: int = 0
    grid_y: int = 0
    ports: list = field(default_factory=list)   # list[RackPortView]


@dataclass
class RackConnectionView:
    src_port_key: str = ""
    dst_port_key: str = ""
    edge_kind: str = "control"
    remove_kind: str = "control"   # control | param | meta
@dataclass
class SystemAudioDevice:
    output_device_name: str = ""
    output_channels:    int = 2
    input_device_name:  str = ""
    input_channels:     int = 0
    export_to_file:     bool = True
    export_sample_rate: int  = 48000
    export_bit_depth:   int  = 24
    # Transient runtime state
    _reported_output_devices: list = field(default_factory=list)
    _reported_input_devices:  list = field(default_factory=list)
    _reported_output_name:    str  = ""
    _reported_input_name:     str  = ""
    _reported_output_hw_channels: int = 0
    _reported_input_hw_channels:  int = 0
    _reported_output_hw_rate: int = 0
    _reported_input_hw_rate:  int = 0
    _preview_backend:         str  = "pygame.mixer"
    _preview_backend_channels:int  = 2
    _input_buffers:           list = field(default_factory=list)  # list[np.ndarray]

    def to_dict(self) -> dict:
        return {
            "output_device_name": self.output_device_name,
            "output_channels":    self.output_channels,
            "input_device_name":  self.input_device_name,
            "input_channels":     self.input_channels,
            "export_to_file":     self.export_to_file,
            "export_sample_rate": self.export_sample_rate,
            "export_bit_depth":   self.export_bit_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemAudioDevice":
        o = cls()
        o.output_device_name = str(d.get("output_device_name", ""))
        o.output_channels    = max(1, int(d.get("output_channels", 2)))
        o.input_device_name  = str(d.get("input_device_name", ""))
        o.input_channels     = max(0, int(d.get("input_channels", 0)))
        o.export_to_file     = bool(d.get("export_to_file", True))
        o.export_sample_rate = int(d.get("export_sample_rate", 48000))
        o.export_bit_depth   = int(d.get("export_bit_depth", 24))
        return o

    @classmethod
    def knobs(cls) -> list["KnobSpec"]:
        return [
            KnobSpec("output_channels",    "Output Ch.", "int",  2, 1, 16, 1, "", [], False, "Routing Device", ".0f"),
            KnobSpec("input_channels",     "Input Ch.",  "int",  0, 0, 16, 1, "", [], False, "Routing Device", ".0f"),
            KnobSpec("export_to_file",     "Render Main","bool", True, 0, 1, 0, "", [], False, "File Render"),
            KnobSpec("export_sample_rate", "Render SR",  "int", 48000, 8000, 384000, 1, "Hz", [], False, "File Render", ".0f"),
            KnobSpec("export_bit_depth",   "Render Bits","int", 24, 16, 32, 1, "", [], False, "File Render", ".0f"),
        ]

    def output_key(self, idx: int) -> str:
        return f"__sys_out_{idx + 1}__"

    def input_key(self, idx: int) -> str:
        return f"__sys_in_{idx + 1}__"

    def output_keys(self) -> list[str]:
        return [self.output_key(i) for i in range(max(1, int(self.output_channels)))]

    def input_keys(self) -> list[str]:
        return [self.input_key(i) for i in range(max(0, int(self.input_channels)))]
@dataclass
class ParamNode:
    """A routing-graph node whose complex output is extracted to a scalar time series
    and written to one or more target voice attributes, enabling smooth parametric modulation.

    Signal flow
    -----------
    1. ParamNode appears in the routing graph as a regular node (zero independent source).
    2. Other nodes feed into it via standard RoutingEdges (weight / angle / delay).
    3. After the routing solve, its complex output is converted to float64 via *extractor*.
    4. The resulting time series is injected as a *param_override* when re-synthesizing
       each target voice, keeping the derivative smooth even under heavy modulation.
    5. Multiple targets can share the same param signal (multi-source → multi-target).

    Extractor choices (mirror routing_engine.ParamEdge extractors)
    --------------------------------------------------------------
    magnitude  |z|   · always positive · good for density / amplitude driving
    real       Re(z) · signed · follows analytic real part
    imag       Im(z) · signed quadrature
    phase      arg(z) · [-π, π] · good for pitch tracking
    energy     |z|²  · heavier weighting of loud moments
    rms        smoothed magnitude (128-sample window)
    """
    ARCHETYPE_KEY: ClassVar[str] = "analytic.param_node"

    key:           str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:         str   = "Param"
    targets:       list  = field(default_factory=list)   # list[{"voice_key": str, "attr": str}]
    extractor:     str   = "magnitude"
    default_value: float = 0.0     # output when no routing edges feed this node
    low:           float = 0.0     # output is clamped to [low, high]
    high:          float = 1.0
    color: list = field(default_factory=lambda: [180, 140, 220])

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"extractor", "low", "high"}),
            tracked_inputs=frozenset({"signal_in"}),
        )

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "targets": list(self.targets),
                "extractor": self.extractor,
                "default_value": self.default_value,
                "low": self.low, "high": self.high, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamNode":
        o = cls.__new__(cls)
        o.key          = d.get("key", uuid.uuid4().hex[:8])
        o.label        = d.get("label", "Param")
        # Migrate legacy single-target fields to the targets list
        if "targets" in d:
            o.targets = [dict(t) for t in d["targets"]]
        else:
            vk = d.get("target_voice_key", "")
            at = d.get("target_attr", "")
            o.targets = [{"voice_key": vk, "attr": at}] if (vk or at) else []
        o.extractor    = d.get("extractor", "magnitude")
        o.default_value = float(d.get("default_value", 0.0))
        o.low          = float(d.get("low", 0.0))
        o.high         = float(d.get("high", 1.0))
        o.color        = d.get("color", [180, 140, 220])
        return o

    _EXTRACTORS = ["magnitude", "real", "imag", "phase", "energy", "rms"]

    @classmethod
    def knobs(cls) -> list:
        """Static knobs — label, extractor, default/clamp.
        Target voice/attr are rendered dynamically by the panel (dynamic dropdown)."""
        return [
            KnobSpec("label",         "Label",     "str",    "Param", 0, 0, 0, "", [], False, "Param Node"),
            KnobSpec("extractor",     "Extractor", "choice", "magnitude", 0, 5, 1, "",
                     cls._EXTRACTORS, False, "Param Node"),
            KnobSpec("default_value", "Default",   "float", 0.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
            KnobSpec("low",           "Min clamp", "float", 0.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
            KnobSpec("high",          "Max clamp", "float", 1.0, -1e4, 1e4, 0, "", [], False, "Param Node"),
        ]


# ---------------------------------------------------------------------------
# ControlSlider / ControlSurface — user-configurable interactive controls
#
# Each ControlSlider is an independent routing-graph DC source node.  Its
# output is a constant complex signal at its current scaled value, so any
# ParamNode that receives it via a RoutingEdge will be driven by the slider.
# Multiple sliders are grouped under one ControlSurface for the UI — the
# surface itself is a voice-list item; its sliders each appear in the routing
# graph individually.
#
# "n-channel of any edge": each slider is its own key in the routing graph,
# so a single surface can fan out N independent DC lanes to N distinct
# ParamNodes (or share them) just by drawing routing edges.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SidecarBus — source-identified per-channel metadata carried alongside
# analytic signals through the synthesis and routing pipeline.
#
# Every synthesis source (voice, LFO, module, control slider) may register
# arbitrary named numpy arrays keyed by (source_key, channel_name).  The bus
# is built during _synthesize_patch and returned to callers that request it
# via the _return_sidecar parameter.
#
# Conventions (not enforced — any channel name is legal):
#   "envelope"   float64   amplitude envelope of a voice, shape (n,), range [0,1]
#   "amplitude"  float64   |z(t)| instantaneous magnitude
#   "phase"      float64   arg(z(t)) instantaneous phase in radians
#   "frequency"  float64   instantaneous frequency in Hz (derived from analytic phase)
#   "value"      float64   DC scalar for a ControlSlider, broadcast to (n,)
#   "routed"     complex128 full routed output X[node] after the routing solve
#
# Usage pattern:
#   bus.put(key, "envelope", env_array)          # populate during synthesis
#   env = bus.get(key, "envelope")               # retrieve later (or None)
#   up  = bus.upstream_of(dst_key, edges)        # view from nodes feeding dst
# ---------------------------------------------------------------------------
@dataclass
class SidecarBus:
    """Carries source-identified, named time-series metadata alongside analytic signals.

    data: dict[source_key: str, dict[channel_name: str, np.ndarray]]
    """
    data: dict = field(default_factory=dict)

    def put(self, source_key: str, channel: str, arr: np.ndarray) -> None:
        """Register a numpy array under (source_key, channel).  Arrays are stored
        by reference — callers should pass already-computed arrays; copying is the
        caller's responsibility when mutation is a concern."""
        if source_key not in self.data:
            self.data[source_key] = {}
        self.data[source_key][channel] = arr

    def get(self, source_key: str, channel: str, n: int = 0) -> "np.ndarray | None":
        """Return the channel array for source_key, or None if absent.
        When *n* > 0 the array is sliced to at most *n* samples."""
        d = self.data.get(source_key)
        if d is None:
            return None
        arr = d.get(channel)
        if arr is None:
            return None
        return arr[:n] if (n > 0 and n < len(arr)) else arr

    def sources(self) -> list:
        """All registered source keys."""
        return list(self.data.keys())

    def channels(self, source_key: str) -> list:
        """Channel names available for *source_key*."""
        return list(self.data.get(source_key, {}).keys())

    def upstream_of(self, dst_key: str, edges: list) -> "SidecarBus":
        """Return a new SidecarBus containing only sidecar from nodes that
        directly feed *dst_key* via a RoutingEdge (single-hop upstream)."""
        src_keys = {e.src_key for e in edges if e.dst_key == dst_key}
        result = SidecarBus()
        for sk in src_keys:
            if sk in self.data:
                result.data[sk] = self.data[sk]
        return result

    def all_upstream_of(self, dst_key: str, edges: list) -> "SidecarBus":
        """Return sidecar from *all* transitive ancestors of *dst_key*
        (BFS through the routing graph).  Useful for inspecting the full
        signal lineage that contributed to a node."""
        visited: set = set()
        queue: list = [dst_key]
        result = SidecarBus()
        edge_map: dict = {}
        for e in edges:
            edge_map.setdefault(e.dst_key, []).append(e.src_key)
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for src in edge_map.get(node, []):
                if src in self.data:
                    result.data[src] = self.data[src]
                queue.append(src)
        return result

    def trim(self, n: int) -> None:
        """Trim every channel array to at most *n* samples in-place."""
        for sk in self.data:
            for ch in list(self.data[sk]):
                arr = self.data[sk][ch]
                if len(arr) > n:
                    self.data[sk][ch] = arr[:n]

    def merge(self, other: "SidecarBus") -> None:
        """Absorb all channels from *other* (last-write-wins on key collision)."""
        for sk, chs in other.data.items():
            if sk not in self.data:
                self.data[sk] = {}
            self.data[sk].update(chs)


@dataclass
class ControlSlider:
    ARCHETYPE_KEY: ClassVar[str] = "analytic.control_slider"

    key:    str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:  str   = "Ctrl"
    value:  float = 0.5    # normalised position [0, 1]
    low:    float = 0.0    # maps value=0 → low
    high:   float = 1.0    # maps value=1 → high
    is_log: bool  = False
    color:  list  = field(default_factory=lambda: [140, 200, 160])

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        # DC source: constant for any block size, no inputs required.
        return BlockFaculty(
            constant_inputs=frozenset(),
            tracked_inputs=frozenset(),
        )

    def scaled_value(self) -> float:
        """Return the actual value in [low, high] from the normalised position."""
        if self.is_log and self.low > 0 and self.high > 0:
            return self.low * (self.high / self.low) ** max(0.0, min(1.0, self.value))
        return self.low + max(0.0, min(1.0, self.value)) * (self.high - self.low)

    @classmethod
    def knobs(cls) -> list:
        return [
            KnobSpec("label",  "Label", "str",   "Ctrl", 0,    0,   0, "",  [], False, "Slider"),
            KnobSpec("low",    "Low",   "float",  0.0, -1e6, 1e6,   0, "",  [], False, "Slider", ".4g"),
            KnobSpec("high",   "High",  "float",  1.0, -1e6, 1e6,   0, "",  [], False, "Slider", ".4g"),
            KnobSpec("is_log", "Log",   "bool",  False, 0,    1,    0, "",  [], False, "Slider"),
            KnobSpec("value",  "Value", "float",  0.5,  0.0,  1.0,  0, "",  [], False, "Slider", ".4f"),
        ]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "value": self.value,
                "low": self.low, "high": self.high,
                "is_log": self.is_log, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSlider":
        o = cls.__new__(cls)
        o.key    = d.get("key",    uuid.uuid4().hex[:8])
        o.label  = d.get("label",  "Ctrl")
        o.value  = float(d.get("value",  0.5))
        o.low    = float(d.get("low",    0.0))
        o.high   = float(d.get("high",   1.0))
        o.is_log = bool(d.get("is_log",  False))
        o.color  = d.get("color", [140, 200, 160])
        return o


@dataclass
class ControlSurface:
    """Named group of ControlSliders; each slider is a routing-graph DC node.

    Selecting a ControlSurface in the voice list shows all its sliders in the
    PartialPanel, making them interactively adjustable in real time.  Adding a
    RoutingEdge from a slider's key to a ParamNode's key lets the slider drive
    any voice attribute continuously.
    """
    ARCHETYPE_KEY: ClassVar[str] = "analytic.control_surface"

    key:     str  = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:   str  = "Control"
    color:   list = field(default_factory=lambda: [100, 180, 140])
    sliders: list = field(default_factory=list)   # list[ControlSlider]

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset(),
            tracked_inputs=frozenset(),
        )

    @classmethod
    def knobs(cls) -> list:
        # Only the surface-level label is a knob; individual sliders are
        # rendered as a bespoke multi-slider UI in PartialPanel.
        return [
            KnobSpec("label", "Label", "str", "Control", 0, 0, 0, "", [], False, "Surface"),
        ]

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "color": self.color,
                "sliders": [s.to_dict() for s in self.sliders]}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSurface":
        o = cls.__new__(cls)
        o.key     = d.get("key",   uuid.uuid4().hex[:8])
        o.label   = d.get("label", "Control")
        o.color   = d.get("color", [100, 180, 140])
        o.sliders = [ControlSlider.from_dict(s) for s in d.get("sliders", [])]
        return o


# ---------------------------------------------------------------------------
# AnalyticModule — a bespoke signal-processing node in the routing graph.
#
# Unlike AnalyticVoice (full synthesis driver) or AnalyticMixer (routing
# aggregator), a Module has a self-contained signal generation algorithm
# selected by *module_type*.  It participates in the routing graph fully —
# receiving and sending analytic signals via RoutingEdges — but its
# independent source signal is produced entirely by its own algorithm.
#
# LFO is the first module type.  LFODefinition remains for backward
# compatibility with saved patches, but new patches should use AnalyticModule
# with module_type="lfo".  A Module LFO gains one capability LFODefinition
# lacks: because it is a proper routing node, signals from other nodes can
# be summed into its output analytically (ring-modulation, sub-mixing).
# To FM its rate, route a voice → ParamNode targeting the module's rate_hz.
#
# Current module types
# --------------------
#   lfo          Analytic LFO oscillator: rate_hz / shape / depth /
#                phase_offset.  Routing inputs add to the output (AM/ring-mod).
#   passthrough  Zero independent source — output is the sum of routing inputs.
#                Useful as a named sub-bus / side-chain point.
# ---------------------------------------------------------------------------
@dataclass
class AnalyticModule:
    ARCHETYPE_KEY: ClassVar[str] = "analytic.module"

    key:         str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label:       str   = "Module"
    module_type: str   = "lfo"
    muted:       bool  = False
    color:       list  = field(default_factory=lambda: [200, 140, 220])
    # LFO parameters (meaningful when module_type == "lfo")
    rate_hz:      float = 1.0
    shape:        str   = "Sine"
    phase_offset: float = 0.0
    depth:        float = 1.0
    # PitchQuantizer parameters (meaningful when module_type == "pitch_quantizer")
    quantizer_scale_degrees: list  = field(default_factory=list)  # [] = use tuning.scale_degrees
    interpolation_mode:      str   = "discrete"   # see QuantizerHandle._INTERP_MODES
    portamento_time:         float = 0.05   # s  — glide time (portamento/spline/legato)
    slew_rate:               float = 100.0  # semitones/s
    slew2_accel:             float = 200.0  # semitones/s²
    spline_tension:          float = 0.5    # Hermite tangent scale
    # Multi-channel LFO (meaningful when module_type == "lfo" and non-empty)
    # Each entry: {scale, amplitude, rate_hz, tension, phase_offset, shape}
    # Empty → legacy single-channel behaviour from rate_hz/shape/phase_offset/depth.
    lfo_channels:            list  = field(default_factory=list)
    # Interaural parameters (meaningful when module_type == "interaural")
    # ch1 params always active; ch2 params used only when ch2 input edges exist
    iau_azimuth:             float = 0.0    # [-1, 1]  left=−1, right=+1
    iau_elevation:           float = 0.0    # [-1, 1]  down=−1, up=+1
    iau_distance:            float = 0.0    # [0, 1]   near=0, far=1
    iau_width:               float = 0.0    # [0, 1]   point=0, spread=1
    iau_azimuth_ch2:         float = 0.0
    iau_elevation_ch2:       float = 0.0
    iau_distance_ch2:        float = 0.0
    iau_width_ch2:           float = 0.0
    # State machine parameters (meaningful when module_type == "state_machine")
    sm_plugin:      str   = ""     # plugin filename stem (no path, no .py)
    sm_n_items:     int   = 1      # number of physics items
    sm_items:       list  = field(default_factory=list)  # item names from plugin
    sm_vars:        list  = field(default_factory=list)  # output var names from plugin
    sm_bundle_ports: list = field(default_factory=list)  # declared wide/bundle ports for graph authoring
    sm_state_vars:  list  = field(default_factory=list)  # persisted scalar state vars from plugin
    sm_params:      dict  = field(default_factory=dict)  # plugin parameter values
    sm_use_torch:   bool  = False  # prefer torch tensors when available
    # Signal layer this SM module operates at.  Determines where in the
    # causal chain it sits:
    #   "performer"  — receives driver outputs (keyed by item_slot), owns
    #                  instrument states, emits per-instrument signals
    #   "room"       — receives instrument outputs, emits mic stream(s)
    # Empty string means unspecified (legacy / backward-compat).
    signal_layer:   str   = ""
    # Transient — not serialized
    _sm_state:     dict  = field(default_factory=dict)  # {item: {var: scalar}}
    _sm_out_cache: dict  = field(default_factory=dict)  # {node_key: complex128 array}
    _sm_log_text:  str   = ""                           # captured plugin log/output
    _sm_aux_state: dict  = field(default_factory=dict)  # plugin-owned transient caches/state

    _MODULE_TYPES = ["lfo", "passthrough", "pitch_quantizer", "interaural", "state_machine"]
    _LFO_SHAPES   = ["Sine", "Triangle", "Sawtooth", "Square"]
    _INTERP_MODES = ["discrete", "portamento", "slew", "slew2", "spline", "legato"]

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        # Conservatively declare only the parameters that are safe to hold
        # constant across a block for any module type. Mode-specific analysis
        # happens at graph build time when the actual module_type is known.
        return BlockFaculty(
            constant_inputs=frozenset({"rate_hz", "depth", "shape", "phase_offset",
                                       "iau_azimuth", "iau_elevation", "iau_distance"}),
            tracked_inputs=frozenset({"signal_in"}),
        )

    @classmethod
    def knobs(cls) -> list:
        _PQ = ("module_type", "pitch_quantizer")
        return [
            KnobSpec("module_type",  "Type",   "choice", "lfo",  0, 3, 1, "",
                     cls._MODULE_TYPES, False, "Module", "", "", True),
            # LFO
            KnobSpec("rate_hz",      "Rate",   "float",  1.0,  0.01, 50.0,      0, "Hz",  [],
                     True,  "LFO", ".3f", "", False, ("module_type", "lfo")),
            KnobSpec("shape",        "Shape",  "choice", "Sine", 0,  3,   1, "",
                     cls._LFO_SHAPES, False, "LFO", "", "", False, ("module_type", "lfo")),
            KnobSpec("phase_offset", "Phase",  "float",  0.0, -math.pi, math.pi, 0, "rad", [],
                     False, "LFO", ".3f", "", False, ("module_type", "lfo")),
            KnobSpec("depth",        "Depth",  "float",  1.0,  0.0,  4.0,  0, "",  [],
                     False, "LFO", ".3f", "", False, ("module_type", "lfo")),
            # PitchQuantizer
            KnobSpec("interpolation_mode", "Interp",   "choice", "discrete", 0, 5, 1, "",
                     cls._INTERP_MODES, False, "PitchQuantizer", "", "", True, _PQ),
            KnobSpec("portamento_time",    "Glide",    "float",  0.05, 0.0,   4.0,  0, "s",  [],
                     True,  "PitchQuantizer", ".3f", "", False, _PQ),
            KnobSpec("slew_rate",          "Slew rate","float",  100.0, 1.0, 1000.0, 0, "st/s", [],
                     False, "PitchQuantizer", ".1f", "", False, _PQ),
            KnobSpec("slew2_accel",        "Accel",    "float",  200.0, 1.0, 5000.0, 0, "st/s²", [],
                     False, "PitchQuantizer", ".1f", "", False, _PQ),
            KnobSpec("spline_tension",     "Tension",  "float",  0.5,  0.0,  2.0,  0, "",  [],
                     False, "PitchQuantizer", ".2f", "", False, _PQ),
            # Interaural — ch1 (always active)
            KnobSpec("iau_azimuth",    "Az ch1",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_elevation",  "El ch1",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_distance",   "Dist ch1", "float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_width",      "Width ch1","float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch1", ".3f", "", False, ("module_type", "interaural")),
            # Interaural — ch2 (active when ch2 input edges exist)
            KnobSpec("iau_azimuth_ch2",  "Az ch2",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_elevation_ch2","El ch2",   "float",  0.0, -1.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_distance_ch2", "Dist ch2", "float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
            KnobSpec("iau_width_ch2",    "Width ch2","float",  0.0,  0.0, 1.0, 0, "", [], False,
                     "Interaural ch2", ".3f", "", False, ("module_type", "interaural")),
        ]

    def ch1_key(self) -> str:
        return f"{self.key}_ch1"

    def ch2_key(self) -> str:
        return f"{self.key}_ch2"

    def lfo_ch_key(self, i: int) -> str:
        return f"{self.key}_lfoch{i}"

    def sm_out_key(self, item: str, var: str) -> str:
        return f"{self.key}_sm_{item}_{var}"

    def sm_out_keys(self) -> list:
        return [self.sm_out_key(item, var)
                for item in self.sm_items for var in self.sm_vars]

    @staticmethod
    def default_lfo_channel() -> dict:
        return {"scale": 0.0, "amplitude": 1.0, "rate_hz": 1.0,
                "tension": 1.0, "phase_offset": 0.0, "shape": "Sine",
                "resample": 1, "slew_order": 1, "slew": 0.0}

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "module_type": self.module_type, "muted": self.muted,
                "color": self.color, "rate_hz": self.rate_hz,
                "shape": self.shape, "phase_offset": self.phase_offset,
                "depth": self.depth,
                "quantizer_scale_degrees": list(self.quantizer_scale_degrees),
                "interpolation_mode":      self.interpolation_mode,
                "portamento_time":         self.portamento_time,
                "slew_rate":               self.slew_rate,
                "slew2_accel":             self.slew2_accel,
                "spline_tension":          self.spline_tension,
                "lfo_channels":            list(self.lfo_channels),
                "iau_azimuth":             self.iau_azimuth,
                "iau_elevation":           self.iau_elevation,
                "iau_distance":            self.iau_distance,
                "iau_width":               self.iau_width,
                "iau_azimuth_ch2":         self.iau_azimuth_ch2,
                "iau_elevation_ch2":       self.iau_elevation_ch2,
                "iau_distance_ch2":        self.iau_distance_ch2,
                "iau_width_ch2":           self.iau_width_ch2,
                "sm_plugin":               self.sm_plugin,
                "sm_n_items":              self.sm_n_items,
                "sm_items":                list(self.sm_items),
                "sm_vars":                 list(self.sm_vars),
                "sm_bundle_ports":         [dict(p) for p in self.sm_bundle_ports],
                "sm_state_vars":           list(self.sm_state_vars),
                "sm_params":               dict(self.sm_params),
                "sm_use_torch":            self.sm_use_torch,
                "signal_layer":            self.signal_layer}

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticModule":
        o = cls.__new__(cls)
        o.key          = d.get("key",         uuid.uuid4().hex[:8])
        o.label        = d.get("label",        "Module")
        o.module_type  = d.get("module_type",  "lfo")
        if o.module_type not in cls._MODULE_TYPES:
            o.module_type = "lfo"
        o.muted        = bool(d.get("muted",   False))
        o.color        = d.get("color",        [200, 140, 220])
        o.rate_hz      = float(d.get("rate_hz",      1.0))
        o.shape        = d.get("shape",        "Sine")
        o.phase_offset = float(d.get("phase_offset", 0.0))
        o.depth        = float(d.get("depth",        1.0))
        o.quantizer_scale_degrees = list(d.get("quantizer_scale_degrees", []))
        o.interpolation_mode      = d.get("interpolation_mode", "discrete")
        if o.interpolation_mode not in cls._INTERP_MODES:
            o.interpolation_mode = "discrete"
        o.portamento_time    = float(d.get("portamento_time", 0.05))
        o.slew_rate          = float(d.get("slew_rate",       100.0))
        o.slew2_accel        = float(d.get("slew2_accel",     200.0))
        o.spline_tension     = float(d.get("spline_tension",  0.5))
        o.lfo_channels       = [dict(c) for c in d.get("lfo_channels", [])]
        o.iau_azimuth        = float(d.get("iau_azimuth",        0.0))
        o.iau_elevation      = float(d.get("iau_elevation",      0.0))
        o.iau_distance       = float(d.get("iau_distance",       0.0))
        o.iau_width          = float(d.get("iau_width",          0.0))
        o.iau_azimuth_ch2    = float(d.get("iau_azimuth_ch2",    0.0))
        o.iau_elevation_ch2  = float(d.get("iau_elevation_ch2",  0.0))
        o.iau_distance_ch2   = float(d.get("iau_distance_ch2",   0.0))
        o.iau_width_ch2      = float(d.get("iau_width_ch2",      0.0))
        o.sm_plugin          = str(d.get("sm_plugin",    ""))
        o.sm_n_items         = int(d.get("sm_n_items",   1))
        o.sm_items           = list(d.get("sm_items",    []))
        o.sm_vars            = list(d.get("sm_vars",     []))
        o.sm_bundle_ports    = [dict(p) for p in d.get("sm_bundle_ports", [])]
        o.sm_state_vars      = list(d.get("sm_state_vars", d.get("sm_vars", [])))
        o.sm_params          = dict(d.get("sm_params",   {}))
        o.sm_use_torch       = bool(d.get("sm_use_torch", False))
        if o.module_type == "state_machine" and o.sm_plugin:
            _plug = _load_sm_plugin(o.sm_plugin)
            if _plug is not None:
                if not o.sm_vars:
                    o.sm_vars = _sm_plugin_output_vars(_plug)
                if not o.sm_state_vars:
                    o.sm_state_vars = _sm_plugin_state_vars(_plug)
                if not o.sm_items:
                    o.sm_items = _sm_plugin_item_names(_plug, o.sm_n_items)
                _defs = _sm_plugin_default_params(_plug)
                o.sm_params = {**_defs, **o.sm_params}
        o._sm_state          = {}
        o._sm_out_cache      = {}
        o._sm_log_text       = ""
        o._sm_aux_state      = {}
        o.signal_layer       = str(d.get("signal_layer", ""))
        return o
class GridViewLayer:
    """Configuration for one visual/interactive layer on the rhythm grid."""

    __slots__ = (
        "name",
        "read_fn",           # (leaf, step_i, extra) -> value
        "bg_fn",             # (value, is_beat, depth, group) -> (r,g,b)
        "brd_fn",            # (value, is_beat, depth, group) -> (r,g,b)
        "label_fn",          # (value) -> (text, (r,g,b)) | None   (left-aligned)
        "label_right_fn",    # (value) -> (text, (r,g,b)) | None   (right-aligned)
        "click_fn",          # (leaf, step_i, extra) -> None   (left click)
        "right_click_fn",    # (cell, lx, ly, pat, div) -> dict | None  (context menu)
        "show_groups",       # bool — render group dots + merge bars
        "show_depth",        # bool — render depth tick marks
    )

    def __init__(
        self,
        name: str,
        *,
        read_fn,
        bg_fn,
        brd_fn,
        label_fn=None,
        label_right_fn=None,
        click_fn=None,
        right_click_fn=None,
        show_groups: bool = False,
        show_depth: bool = False,
    ):
        self.name            = name
        self.read_fn         = read_fn
        self.bg_fn           = bg_fn
        self.brd_fn          = brd_fn
        self.label_fn        = label_fn
        self.label_right_fn  = label_right_fn
        self.click_fn        = click_fn
        self.right_click_fn  = right_click_fn
        self.show_groups     = show_groups
        self.show_depth      = show_depth


@dataclass
class RhythmPattern:
    """Per-bar on/off step grid with per-step velocity and articulation.

    Two representations coexist:
    - Flat (legacy): steps / vel / art lists indexed by integer step.
    - Tree:          beat_nodes (BeatTree) stores the same data with optional
                     per-cell subdivision.  When beat_nodes is not None it is
                     the authoritative representation; the flat lists serve as a
                     projection cache for systems that haven't been updated yet.

    Each pattern also owns independent layer trees for accent and improv.
    These share the same [0,1) bar spine and warp but subdivide independently.

    Use get_tree(div) to obtain a live BeatTree regardless of which mode is
    active.  Use ensure_size(n) as before — it is safe to call on either mode.
    """
    ARCHETYPE_KEY: ClassVar[str] = "analytic.rhythm_pattern"

    name:  str  = "Pat"
    steps: list = field(default_factory=lambda: [False] * 16)
    vel:   list = field(default_factory=lambda: [1.0] * 16)
    art:   list = field(default_factory=lambda: [0] * 16)   # 0=normal 1=staccato 2=legato 3=drone
    # Optional tree representation — None means flat mode
    beat_nodes: "BeatTree | None" = field(default=None, compare=False, repr=False)
    # Independent layer trees (created on demand, same spine, own subdivisions)
    accent_tree: "BeatTree | None" = field(default=None, compare=False, repr=False)
    improv_tree: "BeatTree | None" = field(default=None, compare=False, repr=False)

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"steps", "vel", "art", "beat_nodes",
                                       "accent_tree", "improv_tree"}),
            tracked_inputs=frozenset(),
        )

    # ------------------------------------------------------------------
    # Tree access
    # ------------------------------------------------------------------

    def get_tree(self, div: int = 16) -> "BeatTree":
        """Return the live BeatTree, building from flat if not yet promoted."""
        if self.beat_nodes is not None:
            return self.beat_nodes
        self.ensure_size(div)
        self.beat_nodes = BeatTree.from_flat(self.steps, self.vel, self.art, home_div=div)
        return self.beat_nodes

    def get_accent_tree(self, div: int = 16) -> "BeatTree":
        """Return the accent layer tree, creating a uniform one on first call.

        Accent uses ``vel`` on each leaf as the accent level (0.0 – 2.0).
        """
        if self.accent_tree is not None:
            return self.accent_tree
        self.accent_tree = BeatTree.new_uniform(div, default_on=True, default_vel=1.0)
        return self.accent_tree

    def get_improv_tree(self, div: int = 16) -> "BeatTree":
        """Return the improv-eligibility layer tree, creating uniform on first call.

        Improv uses ``on`` on each leaf as eligibility (True = eligible).
        """
        if self.improv_tree is not None:
            return self.improv_tree
        self.improv_tree = BeatTree.new_uniform(div, default_on=False, default_vel=1.0)
        return self.improv_tree

    def snap_accent_to_rhythm(self) -> None:
        """Restructure accent tree to match the rhythm tree's subdivisions."""
        if self.beat_nodes is None or self.accent_tree is None:
            return
        self.accent_tree.snap_structure_from(self.beat_nodes)

    def snap_improv_to_rhythm(self) -> None:
        """Restructure improv tree to match the rhythm tree's subdivisions."""
        if self.beat_nodes is None or self.improv_tree is None:
            return
        self.improv_tree.snap_structure_from(self.beat_nodes)

    def sync_flat_from_tree(self, div: int | None = None) -> None:
        """Project the tree back onto the flat arrays (for legacy consumers)."""
        if self.beat_nodes is None:
            return
        n = div if div is not None else self.beat_nodes.home_div
        self.ensure_size(n)
        self.steps = self.beat_nodes.steps_array(n)
        self.vel   = self.beat_nodes.vel_array(n)
        self.art   = self.beat_nodes.art_array(n)

    def is_tree_mode(self) -> bool:
        return self.beat_nodes is not None

    # ------------------------------------------------------------------
    # Legacy flat API (unchanged behavior)
    # ------------------------------------------------------------------

    def ensure_size(self, n: int) -> None:
        """Grow steps/vel/art lists to at least n slots."""
        while len(self.steps) < n:
            self.steps.append(False)
        while len(self.vel) < n:
            self.vel.append(1.0)
        while len(self.art) < n:
            self.art.append(0)

    def to_dict(self) -> dict:
        d = {"name": self.name, "steps": list(self.steps),
             "vel": list(self.vel), "art": list(self.art)}
        if self.beat_nodes is not None:
            d["beat_nodes"] = self.beat_nodes.to_dict()
        if self.accent_tree is not None:
            d["accent_tree"] = self.accent_tree.to_dict()
        if self.improv_tree is not None:
            d["improv_tree"] = self.improv_tree.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RhythmPattern":
        rp       = cls()
        rp.name  = d.get("name", "Pat")
        rp.steps = [bool(x) for x in d.get("steps", [])]
        rp.vel   = [float(x) for x in d.get("vel", [])]
        rp.art   = [int(x) for x in d.get("art", [])]
        if "beat_nodes" in d:
            rp.beat_nodes = BeatTree.from_dict(d["beat_nodes"])
        if "accent_tree" in d:
            rp.accent_tree = BeatTree.from_dict(d["accent_tree"])
        if "improv_tree" in d:
            rp.improv_tree = BeatTree.from_dict(d["improv_tree"])
        return rp


@dataclass
class ResolvedNote:
    """Persistent piano-roll note derived from the union score solve."""
    note_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    voice_key: str = ""
    voice_label: str = ""
    layer_key: str = "all"
    start_time: float = 0.0
    duration_s: float = 0.25
    fundamental_hz: float = 440.0
    velocity: float = 1.0
    locked: bool = False
    is_rest: bool = False

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "voice_key": self.voice_key,
            "voice_label": self.voice_label,
            "layer_key": self.layer_key,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "fundamental_hz": self.fundamental_hz,
            "velocity": self.velocity,
            "locked": self.locked,
            "is_rest": self.is_rest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResolvedNote":
        n = cls()
        n.note_id = str(d.get("note_id", n.note_id))
        n.voice_key = str(d.get("voice_key", ""))
        n.voice_label = str(d.get("voice_label", ""))
        n.layer_key = str(d.get("layer_key", "all"))
        n.start_time = float(d.get("start_time", 0.0))
        n.duration_s = float(d.get("duration_s", 0.25))
        n.fundamental_hz = float(d.get("fundamental_hz", 440.0))
        n.velocity = float(d.get("velocity", 1.0))
        n.locked = bool(d.get("locked", False))
        n.is_rest = bool(d.get("is_rest", False))
        return n


@dataclass
class RhythmPage:
    """Complete rhythm configuration for one register-part (e.g. 'bass', 'mid').

    The 'all' page is the global default; per-register pages override it for
    voices that self-declare a matching register.  Any field left at its
    default inherits from the 'all' page at schedule-build time.
    """
    ARCHETYPE_KEY: ClassVar[str] = "analytic.rhythm_page"

    # Core grid
    rhythm_enabled:   bool  = False
    rhythm_division:  int   = 16
    rhythm_patterns:  list  = field(default_factory=lambda: [RhythmPattern(name="Pat 1")])
    rhythm_phrase:    list  = field(default_factory=lambda: [0])
    rhythm_active_pat: int  = 0
    rhythm_swing:     float = 0.0
    rhythm_pocket:    float = 0.0
    rhythm_gate:      float = 0.5
    rhythm_prog_bars: int   = 1
    rhythm_fit_mode:  str   = "drop"
    meter_numerator:  float = 0.0   # 0 = inherit patch meter
    meter_denominator: float = 0.0  # 0 = inherit patch meter
    stress_pattern:   list  = field(default_factory=list)  # e.g. [3, 2]
    # Warp interpolation mode for the beat-tree engine ("linear" | "cosine")
    warp_interpolator: str  = "linear"
    # How the fractional part of meter_numerator is handled:
    #   "warp" — absorbed into the warp curve weighted by stretch (invisible)
    #   "grid" — shown as a visible fractional beat cell in the grid
    frac_beat_mode: str = "warp"
    # Per-module pattern binding: role → pattern index
    # "dynamics" / "improv" / "cadence" → int index into rhythm_patterns
    module_pats: dict = field(default_factory=dict)

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"rhythm_enabled", "rhythm_division",
                                       "rhythm_patterns", "rhythm_phrase",
                                       "rhythm_swing", "rhythm_pocket",
                                       "rhythm_gate", "rhythm_prog_bars",
                                       "rhythm_fit_mode", "meter_numerator",
                                       "meter_denominator", "stress_pattern",
                                       "warp_interpolator", "frac_beat_mode",
                                       "module_pats"}),
            tracked_inputs=frozenset(),
        )

    def to_dict(self) -> dict:
        return {
            "rhythm_enabled":    self.rhythm_enabled,
            "rhythm_division":   self.rhythm_division,
            "rhythm_patterns":   [rp.to_dict() for rp in self.rhythm_patterns],
            "rhythm_phrase":     list(self.rhythm_phrase),
            "rhythm_active_pat": self.rhythm_active_pat,
            "rhythm_swing":      self.rhythm_swing,
            "rhythm_pocket":     self.rhythm_pocket,
            "rhythm_gate":       self.rhythm_gate,
            "rhythm_prog_bars":  self.rhythm_prog_bars,
            "rhythm_fit_mode":   self.rhythm_fit_mode,
            "meter_numerator":   self.meter_numerator,
            "meter_denominator": self.meter_denominator,
            "stress_pattern":    list(self.stress_pattern),
            "warp_interpolator": self.warp_interpolator,
            "frac_beat_mode":   self.frac_beat_mode,
            "module_pats":       dict(self.module_pats),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RhythmPage":
        pg = cls()
        pg.rhythm_enabled    = bool(d.get("rhythm_enabled",   False))
        pg.rhythm_division   = int(d.get("rhythm_division",   16))
        pg.rhythm_patterns   = [RhythmPattern.from_dict(x)
                                 for x in d.get("rhythm_patterns", [])] \
                               or [RhythmPattern(name="Pat 1")]
        pg.rhythm_phrase     = list(d.get("rhythm_phrase",    [0]))
        pg.rhythm_active_pat = int(d.get("rhythm_active_pat", 0))
        pg.rhythm_swing      = float(d.get("rhythm_swing",    0.0))
        pg.rhythm_pocket     = float(d.get("rhythm_pocket",   0.0))
        pg.rhythm_gate       = float(d.get("rhythm_gate",     0.5))
        pg.rhythm_prog_bars  = int(d.get("rhythm_prog_bars",  1))
        pg.rhythm_fit_mode   = str(d.get("rhythm_fit_mode",   "drop"))
        pg.meter_numerator   = max(0.0, float(d.get("meter_numerator", 0.0)))
        pg.meter_denominator = max(0.0, float(d.get("meter_denominator", 0.0)))
        pg.stress_pattern    = [int(max(1, int(x))) for x in d.get("stress_pattern", []) if int(x) > 0]
        pg.warp_interpolator = str(d.get("warp_interpolator", "linear"))
        pg.frac_beat_mode    = str(d.get("frac_beat_mode", "warp"))
        if pg.frac_beat_mode not in ("warp", "grid"):
            pg.frac_beat_mode = "warp"
        pg.module_pats       = dict(d.get("module_pats", {}))
        return pg

    @classmethod
    def from_patch_fields(cls, d: dict) -> "RhythmPage":
        """Build a RhythmPage from a legacy flat-field patch dict."""
        return cls.from_dict(d)


@dataclass
class AnalyticPatch:
    name:       str   = "untitled"
    system_audio: SystemAudioDevice = field(default_factory=SystemAudioDevice)
    voices:     list  = field(default_factory=list)
    lfos:       list  = field(default_factory=list)
    modules:    list  = field(default_factory=list)   # list[AnalyticModule]
    controls:   list  = field(default_factory=list)   # list[ControlSurface]
    mixers:     list  = field(default_factory=lambda: [AnalyticMixer()])  # list[AnalyticMixer]
    param_nodes: list = field(default_factory=list)                        # list[ParamNode]
    _compiled: Any = field(default=None, repr=False, compare=False)
    _compiled_signature: tuple = field(default_factory=tuple, repr=False, compare=False)
    duration:   float = 2.0
    tuning:     "GlobalTuning" = field(default_factory=GlobalTuning)
    preview_sr: int   = 48000
    # Sequence / demo
    seq_scale:           str   = "pentatonic_minor"
    seq_bpm:             float = 120.0
    seq_pattern_idx:     int   = 1     # index into _SEQ_PATTERN_PRESETS
    seq_legato:          float = 0.85
    seq_portamento_s:    float = 0.0   # glide time in seconds
    seq_rubato_shape:    str   = "off"
    seq_rubato_scope:    str   = "bar"
    seq_rubato_amount:   float = 0.0
    seq_octave_span:     int   = 2
    seq_tonic_hz:        float = 440.0 # tonal center / scale root (key being performed)
    meter_numerator:     float = 4.0
    meter_denominator:   float = 4.0
    _seq_note_hz:        float = 0.0  # transient: current note Hz for __patch_seq__ (0 = use seq_tonic_hz)
    seq_bass_octave:     int   = -1   # octave offset for bass-role voices
    seq_root_octave:     int   = -2   # octave offset for root-role voices (pedal)
    seq_stab_octave:     int   =  1   # octave offset for stab-role voices
    seq_chord_prog:      str   = "I_IV_V_I"
    seq_repeats:         int   = 2
    seq_custom_semitones: str  = ""    # e.g. "0,2,4,5,7,9,11" – overrides named scale
    # Projection / output
    projection_mode:        str   = "mono"  # "mono"|"stereo_quadrature"|"stereo_ms"|"lissajous"
    projection_rotation_hz: float = 0.0    # rotates analytic projection plane at this rate
    normalize_output:       bool  = True   # peak-normalize the mix before projection
    performer_phase_mode:   str   = "coherent"  # "coherent" | "individual"
    # Signal routing graph (analytic, pre-projection) — legacy default voice router.
    # New code should use patch.routers[i].graph for named router instances.
    routing: RoutingGraph = field(default_factory=RoutingGraph)
    # Multi-router deployment.  Each RouterInstance owns its own RoutingGraph
    # and is typed by MIXER_LAYERS ("voice_router", "instrument", "master").
    # Empty list means only the legacy `routing` graph is active.
    routers: list = field(default_factory=list)   # list[RouterInstance]
    # UI-only state — not serialized.  When set, only this voice key produces audio.
    solo_key: str | None = None
    resolved_notes: list = field(default_factory=list)  # list[ResolvedNote]
    # Param-series cache — not serialized.  Holds last frame's extracted param
    # node series so _synthesize_patch can apply param overrides in a single
    # solve rather than two.  One-buffer latency; reset on patch load/clear.
    _param_series_cache: dict = field(default_factory=dict)
    # Runtime-only arrangement solve metrics for future placement/allocation UI.
    _arrangement_metrics: dict = field(default_factory=dict)
    # Placement solver output: list of Part objects derived from arrangement metrics.
    parts: list = field(default_factory=list)   # list[Part]
    # Patch-level placement-owned resonator deployment config.
    placement_resonator: PlacementResonatorConfig = field(default_factory=PlacementResonatorConfig)
    # Rhythm programmer
    rhythm_enabled:    bool  = False
    rhythm_division:   int   = 16   # steps per bar: 4 8 12 16 24 32
    rhythm_patterns:   list  = field(default_factory=lambda: [RhythmPattern(name="Pat 1")])
    rhythm_phrase:     list  = field(default_factory=lambda: [0])  # pattern index per bar
    rhythm_active_pat: int   = 0
    rhythm_swing:      float = 0.0  # 0=straight … 0.67=full-triplet push on upbeats
    rhythm_pocket:     float = 0.0  # onset offset in beats (−0.5 … +0.5)
    rhythm_gate:       float = 0.5  # note duration as fraction of step
    rhythm_prog_bars:  int   = 1    # bars one progression cycle spans
    rhythm_fit_mode:   str   = "drop"  # "drop"=truncate to pulses; "extend"=add bars to fit
    stress_pattern:    list  = field(default_factory=list)  # default/all-page stress pattern
    frac_beat_mode:    str   = "warp"  # "warp" | "grid" — how fractional meter is handled
    # Progression probability transforms (0.0 = never, 1.0 = always)
    seq_probabilities: "SequenceProbabilities" = field(
        default_factory=lambda: SequenceProbabilities())
    # Velocity dynamics program (curve + accent grid)
    dynamics_program:  "DynamicsProgram" = field(
        default_factory=lambda: DynamicsProgram())
    dynamics_pages: dict = field(default_factory=dict)  # Dict[str, DynamicsProgram]
    # Stochastic ornament program (grace / chirp / echo)
    improv_program:    "ImprovProgram" = field(
        default_factory=lambda: ImprovProgram())
    improv_pages: dict = field(default_factory=dict)  # Dict[str, ImprovProgram]
    # How score pages combine for a voice. "union" = additive layers,
    # "specific" = only the most-specific layer renders.
    rhythm_layer_mode: str = "union"
    # Per-register rhythm pages.  "all" = global default (mirrors the flat fields
    # above for backward compat).  Additional keys: "bass", "mid", "high".
    rhythm_pages: dict = field(default_factory=dict)  # Dict[str, RhythmPage]
    # UI-only: which page is displayed in the rhythm section (not serialized).
    rhythm_active_page: str = "all"

    def _default_page(self) -> "RhythmPage":
        """Build a RhythmPage that mirrors the current flat rhythm fields."""
        pg = RhythmPage()
        pg.rhythm_enabled    = self.rhythm_enabled
        pg.rhythm_division   = self.rhythm_division
        pg.rhythm_patterns   = self.rhythm_patterns
        pg.rhythm_phrase     = self.rhythm_phrase
        pg.rhythm_active_pat = self.rhythm_active_pat
        pg.rhythm_swing      = self.rhythm_swing
        pg.rhythm_pocket     = self.rhythm_pocket
        pg.rhythm_gate       = self.rhythm_gate
        pg.rhythm_prog_bars  = self.rhythm_prog_bars
        pg.rhythm_fit_mode   = self.rhythm_fit_mode
        pg.meter_numerator   = self.meter_numerator
        pg.meter_denominator = self.meter_denominator
        pg.stress_pattern    = list(self.stress_pattern)
        pg.frac_beat_mode    = self.frac_beat_mode
        return pg

    def page_for(self, register: str) -> "RhythmPage":
        """Return a single RhythmPage for *register* (used by UI page selector).

        The 'all' page is always the live flat-field default.  Named pages are
        stored in rhythm_pages with arbitrary tag-set keys (e.g. 'bass',
        'bass+transient', 'stab+mid').  For UI display of a single named page,
        pass the key directly.
        """
        if register != "all" and register in self.rhythm_pages:
            return self.rhythm_pages[register]
        return self._default_page()

    def dynamics_for(self, key: str) -> "DynamicsProgram":
        if key != "all" and key in self.dynamics_pages:
            return self.dynamics_pages[key]
        return self.dynamics_program

    def improv_for(self, key: str) -> "ImprovProgram":
        if key != "all" and key in self.improv_pages:
            return self.improv_pages[key]
        return self.improv_program

    def beats_per_bar(self) -> float:
        den = max(0.125, float(self.meter_denominator))
        num = max(0.125, float(self.meter_numerator))
        return num * (4.0 / den)

    def page_meter(self, page: "RhythmPage | None" = None) -> tuple[float, float]:
        if page is None:
            return float(self.meter_numerator), float(self.meter_denominator)
        num = float(getattr(page, "meter_numerator", 0.0)) or float(self.meter_numerator)
        den = float(getattr(page, "meter_denominator", 0.0)) or float(self.meter_denominator)
        return max(0.125, num), max(0.125, den)

    def page_beats_per_bar(self, page: "RhythmPage | None" = None) -> float:
        num, den = self.page_meter(page)
        return num * (4.0 / den)

    def bar_duration_s(self) -> float:
        beat_s = 60.0 / max(float(self.seq_bpm), 1.0)
        return beat_s * self.beats_per_bar()

    def ensure_dynamics_page(self, key: str) -> "DynamicsProgram":
        if key == "all":
            return self.dynamics_program
        if key not in self.dynamics_pages:
            self.dynamics_pages[key] = DynamicsProgram.from_dict(self.dynamics_program.to_dict())
        return self.dynamics_pages[key]

    def ensure_improv_page(self, key: str) -> "ImprovProgram":
        if key == "all":
            return self.improv_program
        if key not in self.improv_pages:
            self.improv_pages[key] = ImprovProgram.from_dict(self.improv_program.to_dict())
        return self.improv_pages[key]

    def remove_page(self, register: str) -> None:
        """Delete a named register page from all three page dicts. No-op for 'all'."""
        if register == "all":
            return
        self.rhythm_pages.pop(register, None)
        self.dynamics_pages.pop(register, None)
        self.improv_pages.pop(register, None)
        # Mark patch dirty for UI refresh
        self._arrangement_metrics = {}

    def score_page_items_for_voice(self, voice: "AnalyticVoice") -> "list[tuple[str, RhythmPage]]":
        """Return all score layers that apply to *voice*, ordered least→most specific."""
        voice_tags = {
            getattr(voice, "register",   "all"),
            getattr(voice, "seq_role",   "melody"),
            getattr(voice, "voice_role", "signal"),
            getattr(voice, "key", ""),
        } - {"all", "free", ""}

        stack: list[tuple[int, str, RhythmPage]] = [(0, "all", self._default_page())]
        for key, pg in self.rhythm_pages.items():
            key_tags = {t.strip() for t in key.split("+")} - {"all", ""}
            if not key_tags:
                continue
            if key_tags.issubset(voice_tags):
                stack.append((len(key_tags), key, pg))

        stack.sort(key=lambda x: (x[0], x[1]))
        return [(key, pg) for _, key, pg in stack]

    def score_stack_for_voice(self, voice: "AnalyticVoice") -> "list[RhythmPage]":
        """Return all pages that apply to *voice*, ordered least→most specific.

        A page applies when all of its tag tokens (split on '+') are present in
        the voice's declared tag set (register ∪ seq_role ∪ voice_role).  The
        global default page ('all'/flat fields) is always the base of the stack.
        More-specific pages (more tokens in their key) override it per-step.

        Example: voice has register='bass', seq_role='stab', voice_role='transient'.
          - page key 'bass'              → applies (1 token ⊆ voice tags)
          - page key 'bass+transient'    → applies (2 tokens ⊆ voice tags)
          - page key 'stab+mid'          → does NOT apply ('mid' not in voice tags)
        """
        return [pg for _, pg in self.score_page_items_for_voice(voice)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system_audio": self.system_audio.to_dict(),
            "voices":      [v.to_dict() for v in self.voices],
            "lfos":        [l.to_dict() for l in self.lfos],
            "modules":     [m.to_dict() for m in self.modules],
            "controls":    [cs.to_dict() for cs in self.controls],
            "mixers":      [m.to_dict() for m in self.mixers],
            "param_nodes": [pn.to_dict() for pn in self.param_nodes],
            "duration":   self.duration,
            "tuning":     self.tuning.to_dict(),
            "preview_sr": self.preview_sr,
            "seq_scale":           self.seq_scale,
            "seq_bpm":             self.seq_bpm,
            "seq_pattern_idx":     self.seq_pattern_idx,
            "seq_legato":          self.seq_legato,
            "seq_portamento_s":    self.seq_portamento_s,
            "seq_rubato_shape":    self.seq_rubato_shape,
            "seq_rubato_scope":    self.seq_rubato_scope,
            "seq_rubato_amount":   self.seq_rubato_amount,
            "seq_octave_span":     self.seq_octave_span,
            "seq_tonic_hz":        self.seq_tonic_hz,
            "meter_numerator":     self.meter_numerator,
            "meter_denominator":   self.meter_denominator,
            "seq_bass_octave":     self.seq_bass_octave,
            "seq_root_octave":     self.seq_root_octave,
            "seq_stab_octave":     self.seq_stab_octave,
            "seq_chord_prog":      self.seq_chord_prog,
            "seq_repeats":         self.seq_repeats,
            "seq_custom_semitones": self.seq_custom_semitones,
            "projection_mode":        self.projection_mode,
            "projection_rotation_hz": self.projection_rotation_hz,
            "normalize_output":       self.normalize_output,
            "performer_phase_mode":   self.performer_phase_mode,
            "routing":                self.routing.to_dict(),
            "routers":                [r.to_dict() for r in self.routers],
            # Per-register rhythm pages (excludes "all" — derived from flat fields on load)
            "rhythm_pages":      {k: v.to_dict() for k, v in self.rhythm_pages.items()
                                  if k != "all"},
            # Legacy flat fields — written as mirrors of the "all" page so old
            # readers can still load the patch without the page system.
            "rhythm_enabled":    self.rhythm_enabled,
            "rhythm_division":   self.rhythm_division,
            "rhythm_patterns":   [rp.to_dict() for rp in self.rhythm_patterns],
            "rhythm_phrase":     list(self.rhythm_phrase),
            "rhythm_active_pat": self.rhythm_active_pat,
            "rhythm_swing":      self.rhythm_swing,
            "rhythm_pocket":     self.rhythm_pocket,
            "rhythm_gate":       self.rhythm_gate,
            "rhythm_prog_bars":  self.rhythm_prog_bars,
            "rhythm_fit_mode":   self.rhythm_fit_mode,
            "stress_pattern":    list(self.stress_pattern),
            "frac_beat_mode":   self.frac_beat_mode,
            "seq_probabilities": self.seq_probabilities.to_dict(),
            "dynamics_pages":    {k: v.to_dict() for k, v in self.dynamics_pages.items()
                                  if k != "all"},
            "dynamics_program":  self.dynamics_program.to_dict(),
            "improv_pages":      {k: v.to_dict() for k, v in self.improv_pages.items()
                                  if k != "all"},
            "improv_program":    self.improv_program.to_dict(),
            "rhythm_layer_mode": self.rhythm_layer_mode,
            "resolved_notes":    [n.to_dict() for n in self.resolved_notes],
            "placement_resonator": self.placement_resonator.to_dict(),
            # Placement solver output — persists player counts and hints
            "placement": [
                {
                    "key":          pt.key,
                    "label":        pt.label,
                    "register":     pt.register,
                    "seq_role":     pt.seq_role,
                    "voice_role":   pt.voice_role,
                    "voice_keys":   list(pt.voice_keys),
                    "player_count": pt.player_count,
                    "solver_hints": dict(pt.solver_hints),
                    "chairs": [
                        {
                            "key": ch.key,
                            "label": ch.label,
                            "part_key": ch.part_key,
                            "chair_index": ch.chair_index,
                            "specificity_rank": ch.specificity_rank,
                            "source_voice_keys": list(ch.source_voice_keys),
                            "source_layer_keys": list(ch.source_layer_keys),
                            "performer_count": ch.performer_count,
                            "solver_hints": dict(ch.solver_hints),
                            "performers": [
                                {
                                    "key": pf.key,
                                    "label": pf.label,
                                    "chair_key": pf.chair_key,
                                    "chair_index": pf.chair_index,
                                    "performer_index": pf.performer_index,
                                    "source_voice_keys": list(pf.source_voice_keys),
                                    "source_layer_keys": list(pf.source_layer_keys),
                                    "assigned_note_keys": list(pf.assigned_note_keys),
                                    "body_type": pf.body_type,
                                    "x": pf.x,
                                    "y": pf.y,
                                    "z": pf.z,
                                    "face_x": pf.face_x,
                                    "face_y": pf.face_y,
                                    "face_z": pf.face_z,
                                    "radius": pf.radius,
                                    "angle_deg": pf.angle_deg,
                                    "geometric_delay_ms": pf.geometric_delay_ms,
                                    "humanization_ms": pf.humanization_ms,
                                    "phase_offset_rad": pf.phase_offset_rad,
                                    "gain_db": pf.gain_db,
                                    "pan": pf.pan,
                                }
                                for pf in ch.performers
                            ],
                        }
                        for ch in pt.chairs
                    ],
                }
                for pt in self.parts
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyticPatch":
        p = cls()
        p.name       = d.get("name", "untitled")
        p.system_audio = SystemAudioDevice.from_dict(d.get("system_audio", {}))
        p.voices      = [AnalyticVoice.from_dict(x) for x in d.get("voices", [])]
        p.lfos        = [LFODefinition.from_dict(x) for x in d.get("lfos", [])]
        p.modules     = [AnalyticModule.from_dict(x) for x in d.get("modules", [])]
        p.controls    = [ControlSurface.from_dict(x) for x in d.get("controls", [])]
        raw_mixers    = d.get("mixers", [])
        p.mixers      = [AnalyticMixer.from_dict(x) for x in raw_mixers] \
                        if raw_mixers else [AnalyticMixer()]
        p.param_nodes = [ParamNode.from_dict(x) for x in d.get("param_nodes", [])]
        p.duration   = float(d.get("duration", 2.0))
        _raw_tuning  = d.get("tuning")
        # Legacy patches that predate the tuning block seed from root_hz if present.
        _legacy_root = float(d.get("root_hz", 440.0))
        p.tuning     = GlobalTuning.from_dict(_raw_tuning) if _raw_tuning else GlobalTuning(root_hz=_legacy_root)
        p.preview_sr = int(d.get("preview_sr", 48000))
        p.seq_scale           = d.get("seq_scale", "pentatonic_minor")
        p.seq_bpm             = float(d.get("seq_bpm", 120.0))
        p.seq_pattern_idx     = int(d.get("seq_pattern_idx", 1))
        p.seq_legato          = float(d.get("seq_legato", 0.85))
        p.seq_portamento_s    = float(d.get("seq_portamento_s", 0.0))
        p.seq_rubato_shape    = str(d.get("seq_rubato_shape", "off"))
        if p.seq_rubato_shape not in _SEQ_RUBATO_SHAPES:
            p.seq_rubato_shape = "off"
        p.seq_rubato_scope    = str(d.get("seq_rubato_scope", "bar"))
        if p.seq_rubato_scope not in _SEQ_RUBATO_SCOPES:
            p.seq_rubato_scope = "bar"
        p.seq_rubato_amount   = max(0.0, min(0.95, float(d.get("seq_rubato_amount", 0.0))))
        p.seq_octave_span     = int(d.get("seq_octave_span", 2))
        p.seq_tonic_hz        = float(d.get("seq_tonic_hz", p.tuning.root_hz))
        p.meter_numerator     = max(0.125, float(d.get("meter_numerator", 4.0)))
        p.meter_denominator   = max(0.125, float(d.get("meter_denominator", 4.0)))
        p.seq_bass_octave     = int(d.get("seq_bass_octave", -1))
        p.seq_root_octave     = int(d.get("seq_root_octave", -2))
        p.seq_stab_octave     = int(d.get("seq_stab_octave",  1))
        p.seq_chord_prog      = d.get("seq_chord_prog", "I_IV_V_I")
        p.seq_repeats         = int(d.get("seq_repeats", 2))
        p.seq_custom_semitones = d.get("seq_custom_semitones", "")
        p.projection_mode        = d.get("projection_mode", "mono")
        p.projection_rotation_hz = float(d.get("projection_rotation_hz", 0.0))
        p.normalize_output       = bool(d.get("normalize_output", True))
        p.performer_phase_mode   = str(d.get("performer_phase_mode", "coherent"))
        if p.performer_phase_mode not in ("coherent", "individual"):
            p.performer_phase_mode = "coherent"
        _raw_rp             = d.get("rhythm_patterns", [])
        p.rhythm_enabled    = bool(d.get("rhythm_enabled", False))
        p.rhythm_division   = int(d.get("rhythm_division", 16))
        p.rhythm_patterns   = ([RhythmPattern.from_dict(x) for x in _raw_rp]
                               if _raw_rp else [RhythmPattern(name="Pat 1")])
        p.rhythm_phrase     = list(d.get("rhythm_phrase", [0]))
        p.rhythm_active_pat = int(d.get("rhythm_active_pat", 0))
        p.rhythm_swing      = float(d.get("rhythm_swing", 0.0))
        p.rhythm_pocket     = float(d.get("rhythm_pocket", 0.0))
        p.rhythm_gate       = float(d.get("rhythm_gate", 0.5))
        p.rhythm_prog_bars  = int(d.get("rhythm_prog_bars", 1))
        p.rhythm_fit_mode   = str(d.get("rhythm_fit_mode", "drop"))
        p.stress_pattern    = [int(max(1, int(x))) for x in d.get("stress_pattern", []) if int(x) > 0]
        p.frac_beat_mode    = str(d.get("frac_beat_mode", "warp"))
        if p.frac_beat_mode not in ("warp", "grid"):
            p.frac_beat_mode = "warp"
        # Probabilities — support old flat keys for backward compat with saved patches
        _raw_prob = d.get("seq_probabilities")
        if _raw_prob and isinstance(_raw_prob, dict):
            p.seq_probabilities = SequenceProbabilities.from_dict(_raw_prob)
        else:
            sp = SequenceProbabilities()
            sp.double_back = float(d.get("prob_double_back", 0.0))
            sp.subversion  = float(d.get("prob_subversion",  0.0))
            sp.chromatic   = float(d.get("prob_chromatic",   0.0))
            sp.modal       = float(d.get("prob_modal",       0.0))
            p.seq_probabilities = sp
        _raw_dyn = d.get("dynamics_program")
        if _raw_dyn and isinstance(_raw_dyn, dict):
            p.dynamics_program = DynamicsProgram.from_dict(_raw_dyn)
        raw_dyn_pages = d.get("dynamics_pages", {})
        p.dynamics_pages = {}
        for pg_key, pg_dict in raw_dyn_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.dynamics_pages[pg_key] = DynamicsProgram.from_dict(pg_dict)
        _raw_imp = d.get("improv_program")
        if _raw_imp and isinstance(_raw_imp, dict):
            p.improv_program = ImprovProgram.from_dict(_raw_imp)
        raw_imp_pages = d.get("improv_pages", {})
        p.improv_pages = {}
        for pg_key, pg_dict in raw_imp_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.improv_pages[pg_key] = ImprovProgram.from_dict(pg_dict)
        p.rhythm_layer_mode = str(d.get("rhythm_layer_mode", "union"))
        if p.rhythm_layer_mode not in {"union", "specific"}:
            p.rhythm_layer_mode = "union"
        p.placement_resonator = PlacementResonatorConfig.from_dict(d.get("placement_resonator", {}))
        p.resolved_notes = [ResolvedNote.from_dict(x)
                            for x in d.get("resolved_notes", [])
                            if isinstance(x, dict)]
        # Placement: restore Part list; drop stale keys not matching saved entry
        _placement_raw = d.get("placement", [])
        if _placement_raw:
            p.parts = [
                Part(
                    key=          pr.get("key", ""),
                    label=        pr.get("label", ""),
                    register=     pr.get("register", "all"),
                    seq_role=     pr.get("seq_role", ""),
                    voice_role=   pr.get("voice_role", ""),
                    voice_keys=   list(pr.get("voice_keys", [])),
                    player_count= int(pr.get("player_count", 1)),
                    solver_hints= dict(pr.get("solver_hints", {})),
                    chairs=[
                        Chair(
                            key=ch.get("key", ""),
                            label=ch.get("label", ""),
                            part_key=ch.get("part_key", pr.get("key", "")),
                            chair_index=int(ch.get("chair_index", 1)),
                            specificity_rank=int(ch.get("specificity_rank", 0)),
                            source_voice_keys=list(ch.get("source_voice_keys", [])),
                            source_layer_keys=list(ch.get("source_layer_keys", [])),
                            performer_count=max(1, int(ch.get("performer_count", 1))),
                            solver_hints=dict(ch.get("solver_hints", {})),
                            performers=[
                                PerformerPlacement(
                                    key=pf.get("key", ""),
                                    label=pf.get("label", ""),
                                    chair_key=pf.get("chair_key", ch.get("key", "")),
                                    chair_index=int(pf.get("chair_index", ch.get("chair_index", 1))),
                                    performer_index=int(pf.get("performer_index", 1)),
                                    source_voice_keys=list(pf.get("source_voice_keys", [])),
                                    source_layer_keys=list(pf.get("source_layer_keys", [])),
                                    assigned_note_keys=list(pf.get("assigned_note_keys", [])),
                                    body_type=str(pf.get("body_type", "direct") or "direct"),
                                    x=float(pf.get("x", 0.0)),
                                    y=float(pf.get("y", 0.0)),
                                    z=float(pf.get("z", 1.1)),
                                    face_x=float(pf.get("face_x", 0.0)),
                                    face_y=float(pf.get("face_y", -1.0)),
                                    face_z=float(pf.get("face_z", 0.0)),
                                    radius=float(pf.get("radius", 0.0)),
                                    angle_deg=float(pf.get("angle_deg", 0.0)),
                                    geometric_delay_ms=float(pf.get("geometric_delay_ms", 0.0)),
                                    humanization_ms=float(pf.get("humanization_ms", 0.0)),
                                    phase_offset_rad=float(pf.get("phase_offset_rad", 0.0)),
                                    gain_db=float(pf.get("gain_db", 0.0)),
                                    pan=float(pf.get("pan", 0.0)),
                                )
                                for pf in ch.get("performers", [])
                                if isinstance(pf, dict)
                            ],
                        )
                        for ch in pr.get("chairs", [])
                        if isinstance(ch, dict)
                    ],
                )
                for pr in _placement_raw
                if isinstance(pr, dict) and pr.get("key")
            ]
            if p.parts:
                _refresh_part_placement_layout(p)
        # Rhythm pages — load per-register pages only; "all" is always live from flat fields.
        raw_pages = d.get("rhythm_pages", {})
        p.rhythm_pages = {}
        for pg_key, pg_dict in raw_pages.items():
            if pg_key != "all" and isinstance(pg_dict, dict):
                p.rhythm_pages[pg_key] = RhythmPage.from_dict(pg_dict)
        p.rhythm_active_page = "all"   # always start on the default page
        if "routing" in d:
            p.routing = RoutingGraph.from_dict(d["routing"])
            # H1 fix: prune stale edges whose node keys no longer exist in the patch
            valid_keys = _patch_node_keys(p)
            p.routing.prune_keys(valid_keys)
        else:
            p.routing = RoutingGraph()
        # Multi-router instances (new; absent in legacy patches)
        p.routers = [RouterInstance.from_dict(r) for r in d.get("routers", [])]
        p._param_series_cache = {}   # always start fresh on load
        return p

    @classmethod
    def knobs(cls) -> list[KnobSpec]:
        _PROJ    = ["mono", "stereo_quadrature", "stereo_ms", "lissajous"]
        _SR      = [8000.0, 22050.0, 44100.0, 48000.0, 96000.0, 192000.0]
        _PRESETS = list(GLOBAL_TUNING_PRESETS.keys())
        _TEMPS   = GlobalTuning._TEMPERAMENT_CHOICES
        _PHASE   = ["coherent", "individual"]
        return [
            # Global
            KnobSpec("preview_sr",           "Sample rate", "int",   48000, 8000, 192000, 0, "Hz", [], True,  "Global", ".0f", "", True),
            KnobSpec("duration",             "Duration",    "float", 2.0,   0.1,  60.0,   0, "s",  [], False, "Global", ".2f", "", True),
            # Projection
            KnobSpec("projection_mode",        "Proj mode",  "choice", "mono", 0, 3, 1, "", _PROJ, False, "Projection", ".0f", "", True),
            KnobSpec("projection_rotation_hz", "Rot Hz",     "float",  0.0, -200.0, 200.0, 0, "Hz", [], False, "Projection", ".2f"),
            KnobSpec("normalize_output",       "Normalize",  "bool",   True, 0, 1, 1, "",  [], False, "Projection", ""),
            KnobSpec("performer_phase_mode",   "Perf phase", "choice", "coherent", 0, 1, 1, "", _PHASE, False, "Projection"),
            # Tuning
            KnobSpec("tuning.preset_name",   "Preset",      "choice", "a440_12tet", 0,
                     max(0, len(_PRESETS) - 1), 1, "", _PRESETS, False, "Tuning", "", "", True),
            KnobSpec("tuning.root_hz",       "Tuning root", "float",  440.0, 20.0, 8000.0, 0, "Hz", [], True, "Tuning", ".2f"),
            KnobSpec("tuning.temperament",   "Temperament", "choice", "12tet", 0,
                     max(0, len(_TEMPS) - 1), 1, "", _TEMPS, False, "Tuning", "", "", True),
            KnobSpec("tuning.scale_name",    "Scale",       "str",    "chromatic", 0, 0, 0, "", [], False, "Tuning"),
        ]

    @staticmethod
    def default_patch() -> "AnalyticPatch":
        p = AnalyticPatch()
        p.name = "default"
        for i, (hz, col) in enumerate([
            (440.0, [100, 160, 255]),
            (880.0, [255, 130, 60]),
            (220.0, [120, 220, 120]),
        ]):
            voice = AnalyticVoice()
            voice.label   = f"V{i+1}"
            voice.freq_hz = hz
            voice.color   = col
            p.voices.append(voice)
        lfo = LFODefinition()
        lfo.label = "LFO1"
        lfo.rate_hz = 2.5
        p.lfos.append(lfo)
        return p


# Lazy bridges keep the data model independent from routing/score/synthesis modules
# while preserving legacy classmethod behavior during patch loading.
def _patch_node_keys(*args, **kwargs):
    from analytic_routing import _patch_node_keys as _impl
    return _impl(*args, **kwargs)


def _refresh_part_placement_layout(*args, **kwargs):
    from analytic_score import _refresh_part_placement_layout as _impl
    return _impl(*args, **kwargs)


def _load_sm_plugin(*args, **kwargs):
    from analytic_synth_legacy import _load_sm_plugin as _impl
    return _impl(*args, **kwargs)


def _sm_plugin_output_vars(*args, **kwargs):
    from analytic_synth_legacy import _sm_plugin_output_vars as _impl
    return _impl(*args, **kwargs)


def _sm_plugin_state_vars(*args, **kwargs):
    from analytic_synth_legacy import _sm_plugin_state_vars as _impl
    return _impl(*args, **kwargs)


def _sm_plugin_item_names(*args, **kwargs):
    from analytic_synth_legacy import _sm_plugin_item_names as _impl
    return _impl(*args, **kwargs)


def _sm_plugin_default_params(*args, **kwargs):
    from analytic_synth_legacy import _sm_plugin_default_params as _impl
    return _impl(*args, **kwargs)
