"""sequence_engine.py

Assembles sequences of analytic notes (rendered by signal_generator_v2) into
a single phase-continuous global AdaptiveSampleBuffer that can be projected
to audio with any of the numpy / torch projection paths.

Core pipeline
─────────────
ArpeggioRule ──► NoteSchedule ──► SequenceRenderer ──► AdaptiveSampleBuffer
                                          │
                                   LatticeBuilder   (one driver per note)
                                   PhaseHandoff     (terminal → seed phase)
                                   TimbrePreset     (shared timbral params)

MIDI (future)
─────────────
MidiAdapter.load(path) → NoteSchedule    (requires:  pip install mido)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from graph_solver import BlockFaculty
from signal_generator_v2 import (
    ADSREnvelope,
    AdaptiveSampleBuffer,
    AmplitudeEnvelopeShaper,
    ComplexEnvelope,
    ConstantPhasePath,
    DensityPolicy,
    DriftModel,
    EmissionRange,
    ExponentialEnvelope,
    HarmonicLattice,
    LatticeVoice,
    NullDriftModel,
    PhaseWarpedManifold,
    PiecewiseLinearField,
    PiecewiseMonotoneField,
    PiecewiseSplineField,
    ProjectionPolicy,
    PureSineManifold,
    SigmoidField,
    SmoothSinusoidalDriftModel,
    WaveformManifold,
    WitnessAwareSynthDriver,
    WitnessThresholds,
    PI2,
)


# ============================================================
# Modal scale maps  (semitone intervals from root, one octave)
# ============================================================

MODAL_SCALES: Dict[str, List[int]] = {
    # Church / diatonic modes
    "ionian":               [0, 2, 4, 5, 7, 9, 11],   # major
    "dorian":               [0, 2, 3, 5, 7, 9, 10],
    "phrygian":             [0, 1, 3, 5, 7, 8, 10],
    "lydian":               [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":           [0, 2, 4, 5, 7, 9, 10],
    "aeolian":              [0, 2, 3, 5, 7, 8, 10],   # natural minor
    "locrian":              [0, 1, 3, 5, 6, 8, 10],
    # Harmonic & melodic minor family
    "harmonic_minor":       [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":        [0, 2, 3, 5, 7, 9, 11],
    "phrygian_dominant":    [0, 1, 4, 5, 7, 8, 10],   # V of harmonic minor
    "lydian_dominant":      [0, 2, 4, 6, 7, 9, 10],   # acoustic / Bartók
    # Pentatonic & blues
    "pentatonic_major":     [0, 2, 4, 7, 9],
    "pentatonic_minor":     [0, 3, 5, 7, 10],
    "blues":                [0, 3, 5, 6, 7, 10],
    # Symmetric / exotic
    "whole_tone":           [0, 2, 4, 6, 8, 10],
    "octatonic_hw":         [0, 1, 3, 4, 6, 7, 9, 10],  # half-whole diminished
    "octatonic_wh":         [0, 2, 3, 5, 6, 8, 9, 11],  # whole-half diminished
    "chromatic":            list(range(12)),
    # World / character scales
    "double_harmonic":      [0, 1, 4, 5, 7, 8, 11],   # Byzantine / Arabic
    "hungarian_minor":      [0, 2, 3, 6, 7, 8, 11],
    "neapolitan_minor":     [0, 1, 3, 5, 7, 8, 11],
    "enigmatic":            [0, 1, 4, 6, 8, 10, 11],
    # Jazz / bebop
    "bebop_dominant":       [0, 2, 4, 5, 7, 9, 10, 11],
    "altered":              [0, 1, 3, 4, 6, 8, 10],   # super-locrian
}


# ============================================================
# Scale mood metadata
# ============================================================
# Each entry is a dict with:
#   valence   : float  -1.0 (negative) … +1.0 (positive)
#   arousal   : float  -1.0 (calm/still) … +1.0 (excited/tense)
#   adjectives: List[str]  — descriptive mood words
#   origins   : str   — brief historical/cultural note
#   bpm_range : tuple[int, int]  — suggested tempo range
# Sources: Aristotle's Politics VIII (modes); Guido d'Arezzo / Adam of Fulda
# affect table (Wikipedia Mode article); Schubart Ideen zu einer Ästhetik der
# Tonkunst (1806) key characters; Russell (1980) circumplex valence/arousal model;
# common practice in film scoring and jazz theory.

SCALE_MOODS: Dict[str, Dict] = {
    "ionian": {
        "valence": 0.8, "arousal": 0.3,
        "adjectives": ["bright", "confident", "joyful", "resolved", "triumphant"],
        "origins": "Major scale; Glareanus Ionian (1547). Guido: 'happy'.",
        "bpm_range": (80, 160),
    },
    "dorian": {
        "valence": 0.1, "arousal": 0.2,
        "adjectives": ["serious", "noble", "melancholic-yet-hopeful", "cool", "modal"],
        "origins": "Aristotle: 'moderate and firm'. Adam of Fulda: 'any feeling'. "
                   "Prevalent in Celtic folk, jazz, and Gregorian chant.",
        "bpm_range": (60, 140),
    },
    "phrygian": {
        "valence": -0.5, "arousal": 0.6,
        "adjectives": ["dark", "tense", "exotic", "mysterious", "inciting", "Spanish"],
        "origins": "Aristotle: 'ecstatic excitement'. Adam of Fulda: 'vehement'. "
                   "Foundation of Flamenco; evokes urgency and the orient.",
        "bpm_range": (60, 130),
    },
    "lydian": {
        "valence": 0.9, "arousal": 0.5,
        "adjectives": ["dreamy", "ethereal", "floating", "otherworldly", "bright"],
        "origins": "Guido d'Arezzo: 'happy'. Plato warned it softens character. "
                   "Used in film scoring for wonder and the fantastic.",
        "bpm_range": (70, 160),
    },
    "mixolydian": {
        "valence": 0.5, "arousal": 0.4,
        "adjectives": ["bluesy", "nostalgic", "laid-back", "earthy", "folk-like"],
        "origins": "Aristotle: 'grief and anxiety' (ancient context). Modern: "
                   "rock, folk, Celtic. Dominant seventh gives unresolved tension.",
        "bpm_range": (70, 150),
    },
    "aeolian": {
        "valence": -0.5, "arousal": -0.1,
        "adjectives": ["sad", "introspective", "longing", "dark", "natural"],
        "origins": "Natural minor. Guido Hypodorian: 'sad'. Most common minor mode "
                   "in Western music; emotive backbone of ballads.",
        "bpm_range": (50, 130),
    },
    "locrian": {
        "valence": -0.9, "arousal": 0.7,
        "adjectives": ["unstable", "dissonant", "ominous", "theoretical", "tense"],
        "origins": "Diminished tonic triad makes tonal center unstable. Rarely used "
                   "melodically; appears in metal and avant-garde contexts.",
        "bpm_range": (60, 140),
    },
    "harmonic_minor": {
        "valence": -0.4, "arousal": 0.5,
        "adjectives": ["dramatic", "exotic", "gothic", "passionate", "operatic"],
        "origins": "Raised 7th creates leading tone and augmented 2nd. "
                   "Classical and Romantic drama; Middle-Eastern color.",
        "bpm_range": (55, 140),
    },
    "melodic_minor": {
        "valence": -0.1, "arousal": 0.3,
        "adjectives": ["elegant", "sophisticated", "bittersweet", "jazzy"],
        "origins": "Ascending form smooths the aug 2nd of harmonic minor. "
                   "Foundation of modern jazz harmony (Levine 1995).",
        "bpm_range": (60, 160),
    },
    "phrygian_dominant": {
        "valence": -0.6, "arousal": 0.7,
        "adjectives": ["intense", "exotic", "flamenco", "fiery", "Middle-Eastern"],
        "origins": "V mode of harmonic minor. Dominant function with flat-2 "
                   "gives strong tension. Core of Flamenco and klezmer.",
        "bpm_range": (70, 160),
    },
    "lydian_dominant": {
        "valence": 0.3, "arousal": 0.6,
        "adjectives": ["ambiguous", "cinematic", "spacious", "unresolved", "complex"],
        "origins": "Acoustic/Bartók scale. Raised 4 + flat 7 floats between "
                   "major and dominant. Used in film scores for tension.",
        "bpm_range": (60, 140),
    },
    "pentatonic_major": {
        "valence": 0.7, "arousal": 0.2,
        "adjectives": ["open", "pure", "folk", "cheerful", "accessible"],
        "origins": "Universal folk scale. No semitones give consonant stability. "
                   "East Asian traditional music and Western country/blues.",
        "bpm_range": (70, 160),
    },
    "pentatonic_minor": {
        "valence": -0.2, "arousal": 0.3,
        "adjectives": ["soulful", "bluesy", "raw", "earthy", "expressive"],
        "origins": "Foundation of blues, rock, and much of popular music. "
                   "Omitting semitones keeps it flexible and emotionally direct.",
        "bpm_range": (60, 150),
    },
    "blues": {
        "valence": -0.1, "arousal": 0.5,
        "adjectives": ["gritty", "expressive", "soulful", "bent", "cathartic"],
        "origins": "Pentatonic minor + flat-5 'blue note'. African-American "
                   "musical tradition; basis of jazz and rock.",
        "bpm_range": (55, 150),
    },
    "whole_tone": {
        "valence": 0.0, "arousal": 0.2,
        "adjectives": ["dreamy", "hazy", "impressionist", "floating", "suspended"],
        "origins": "All whole steps — no tonal gravity. Debussy used it to "
                   "evoke water, mist, and ambiguity.",
        "bpm_range": (50, 110),
    },
    "octatonic_hw": {
        "valence": -0.3, "arousal": 0.7,
        "adjectives": ["tense", "angular", "jazzy", "unstable", "chromatic"],
        "origins": "Half-whole diminished scale. Jazz dominant substitutions, "
                   "Messiaen Mode 2. Stravinsky used it extensively.",
        "bpm_range": (70, 180),
    },
    "octatonic_wh": {
        "valence": -0.4, "arousal": 0.5,
        "adjectives": ["mysterious", "symmetric", "dark", "chromatic", "modal"],
        "origins": "Whole-half diminished. Used over diminished chords and "
                   "as a color scale in post-Romantic harmony.",
        "bpm_range": (60, 140),
    },
    "chromatic": {
        "valence": 0.0, "arousal": 0.4,
        "adjectives": ["atonal", "complex", "chromatic", "dense", "non-diatonic"],
        "origins": "All 12 semitones. No inherent mood; context-dependent. "
                   "Used in serial music, dense chromatic passages.",
        "bpm_range": (40, 200),
    },
    "double_harmonic": {
        "valence": -0.6, "arousal": 0.6,
        "adjectives": ["exotic", "Byzantine", "Arabic", "mysterious", "dark"],
        "origins": "Two augmented seconds give a Middle-Eastern character. "
                   "Also called Byzantine scale or Arabic maqam Hijaz.",
        "bpm_range": (55, 130),
    },
    "hungarian_minor": {
        "valence": -0.5, "arousal": 0.5,
        "adjectives": ["dramatic", "Eastern-European", "passionate", "exotic"],
        "origins": "Harmonic minor with raised 4th. Liszt used it to evoke "
                   "Hungarian folk music; Romani-influenced.",
        "bpm_range": (55, 140),
    },
    "neapolitan_minor": {
        "valence": -0.7, "arousal": 0.3,
        "adjectives": ["sorrowful", "dark", "Romantic", "elegiac", "operatic"],
        "origins": "Flat 2 gives a sighing quality. Used in Neapolitan opera "
                   "and Romantic-era piano music for deep pathos.",
        "bpm_range": (45, 110),
    },
    "enigmatic": {
        "valence": 0.0, "arousal": 0.4,
        "adjectives": ["mysterious", "verdi", "puzzling", "chromatic", "rare"],
        "origins": "Verdi's 'scala enigmatica' (1888). Unusual intervals create "
                   "a sense of puzzle and esoteric strangeness.",
        "bpm_range": (50, 120),
    },
    "bebop_dominant": {
        "valence": 0.4, "arousal": 0.8,
        "adjectives": ["swinging", "jazzy", "energetic", "chromatic", "urban"],
        "origins": "Dominant scale + chromatic passing tone aligns chord tones "
                   "with strong beats. Central to bebop improvisation.",
        "bpm_range": (120, 260),
    },
    "altered": {
        "valence": -0.2, "arousal": 0.8,
        "adjectives": ["tense", "dissonant", "modern-jazz", "complex", "unresolved"],
        "origins": "Super-locrian / altered dominant. All extensions are altered "
                   "(b9, #9, b5, #5). Maximum tension before resolution.",
        "bpm_range": (100, 240),
    },
}


# ============================================================
# Chord progressions
# ============================================================
# Each chord is a STRING in Roman numeral notation.
# Format rules (extend as needed):
#   "I"      uppercase = major chord on that degree
#   "i"      lowercase = minor chord on that degree
#   "iv"     minor subdominant
#   "+IV"    prefix '+' = raised/augmented variant
#   "-VII"   prefix '-' = lowered/flat variant (borrowed)
#   "II7"    suffix digit = added 7th (etc.)
#   "V/V"    secondary dominant notation preserved as string
# Keys map to dicts with "chords" (List[str]) and mood metadata.

CHORD_PROGRESSIONS: Dict[str, Dict] = {
    # ── Major / Ionian ──────────────────────────────────────────
    "I_IV_V_I": {
        "chords": ["I", "IV", "V", "I"],
        "mood": "triumphant, resolved, folk",
        "valence": 0.8, "arousal": 0.4,
        "typical_scales": ["ionian", "mixolydian", "pentatonic_major"],
    },
    "I_V_vi_IV": {
        "chords": ["I", "V", "vi", "IV"],
        "mood": "pop anthem, emotional, bittersweet",
        "valence": 0.5, "arousal": 0.4,
        "typical_scales": ["ionian", "pentatonic_major"],
    },
    "I_vi_IV_V": {
        "chords": ["I", "vi", "IV", "V"],
        "mood": "nostalgic, romantic, doo-wop",
        "valence": 0.4, "arousal": 0.2,
        "typical_scales": ["ionian"],
    },
    "ii_V_I": {
        "chords": ["ii", "V", "I"],
        "mood": "jazz resolution, sophisticated, forward motion",
        "valence": 0.6, "arousal": 0.5,
        "typical_scales": ["ionian", "melodic_minor", "bebop_dominant"],
    },
    "ii_V_I_extended": {
        "chords": ["ii7", "V7", "Imaj7", "Imaj7"],
        "mood": "jazz, lush, resolved",
        "valence": 0.7, "arousal": 0.4,
        "typical_scales": ["ionian", "melodic_minor"],
    },
    "I_IV_viio_iii_vi_ii_V_I": {
        "chords": ["I", "IV", "viio", "iii", "vi", "ii", "V", "I"],
        "mood": "circle of fifths, classical, flowing",
        "valence": 0.6, "arousal": 0.3,
        "typical_scales": ["ionian"],
    },
    # ── Minor / Aeolian ──────────────────────────────────────────
    "i_VII_VI_V": {
        "chords": ["i", "-VII", "-VI", "V"],
        "mood": "andalusian, descending, dramatic, fatalistic",
        "valence": -0.5, "arousal": 0.6,
        "typical_scales": ["aeolian", "phrygian", "phrygian_dominant"],
    },
    "i_VI_III_VII": {
        "chords": ["i", "-VI", "-III", "-VII"],
        "mood": "dark epic, cinematic minor, powerful",
        "valence": -0.3, "arousal": 0.6,
        "typical_scales": ["aeolian", "harmonic_minor"],
    },
    "i_iv_V_i": {
        "chords": ["i", "iv", "V", "i"],
        "mood": "classical minor, tense, resolving",
        "valence": -0.3, "arousal": 0.5,
        "typical_scales": ["harmonic_minor", "aeolian"],
    },
    "i_iv_i_V": {
        "chords": ["i", "iv", "i", "V"],
        "mood": "modal minor, folk, unresolved tension",
        "valence": -0.4, "arousal": 0.3,
        "typical_scales": ["aeolian", "dorian"],
    },
    # ── Dorian ──────────────────────────────────────────────────
    "i_IV_i_IV": {
        "chords": ["i", "IV", "i", "IV"],
        "mood": "dorian vamp, funk, groove-oriented",
        "valence": 0.1, "arousal": 0.6,
        "typical_scales": ["dorian", "pentatonic_minor"],
    },
    "i_ii_i_ii": {
        "chords": ["i", "ii", "i", "ii"],
        "mood": "dorian shimmer, cool, jazz-influenced",
        "valence": 0.2, "arousal": 0.3,
        "typical_scales": ["dorian"],
    },
    # ── Phrygian / Flamenco ──────────────────────────────────────
    "i_bII_i_bII": {
        "chords": ["i", "-II", "i", "-II"],
        "mood": "phrygian vamp, flamenco, dark exotic",
        "valence": -0.6, "arousal": 0.7,
        "typical_scales": ["phrygian", "phrygian_dominant"],
    },
    "i_bII_bIII_bII": {
        "chords": ["i", "-II", "-III", "-II"],
        "mood": "flamenco, Spanish, passionate",
        "valence": -0.4, "arousal": 0.8,
        "typical_scales": ["phrygian_dominant", "phrygian"],
    },
    # ── Blues ────────────────────────────────────────────────────
    "blues_12bar": {
        "chords": ["I7", "I7", "I7", "I7",
                   "IV7", "IV7", "I7", "I7",
                   "V7", "IV7", "I7", "V7"],
        "mood": "blues, cathartic, soulful, raw",
        "valence": 0.1, "arousal": 0.6,
        "typical_scales": ["blues", "pentatonic_minor", "mixolydian"],
    },
    "minor_blues_12bar": {
        "chords": ["i7", "i7", "i7", "i7",
                   "iv7", "iv7", "i7", "i7",
                   "-VI7", "V7", "i7", "V7"],
        "mood": "minor blues, dark, expressive",
        "valence": -0.3, "arousal": 0.6,
        "typical_scales": ["blues", "pentatonic_minor"],
    },
    # ── Lydian / Floating ────────────────────────────────────────
    "I_II_I_II": {
        "chords": ["I", "II", "I", "II"],
        "mood": "lydian shimmer, cinematic wonder, floating",
        "valence": 0.8, "arousal": 0.3,
        "typical_scales": ["lydian", "lydian_dominant"],
    },
    # ── Modal / Mixolydian ───────────────────────────────────────
    "I_bVII_IV_I": {
        "chords": ["I", "-VII", "IV", "I"],
        "mood": "mixolydian rock, open, folk-rock",
        "valence": 0.5, "arousal": 0.5,
        "typical_scales": ["mixolydian", "pentatonic_major"],
    },
    # ── Jazz / Altered ───────────────────────────────────────────
    "iii_VI_ii_V": {
        "chords": ["iii", "VI", "ii", "V"],
        "mood": "jazz turnaround, sophisticated, forward",
        "valence": 0.3, "arousal": 0.5,
        "typical_scales": ["ionian", "melodic_minor", "bebop_dominant"],
    },
    "I_III7_VI7_II7_V7": {
        "chords": ["I", "III7", "VI7", "II7", "V7"],
        "mood": "ragtime / rhythm changes bridge, swinging",
        "valence": 0.5, "arousal": 0.7,
        "typical_scales": ["ionian", "bebop_dominant"],
    },
    # ── Chromatic / Dramatic ─────────────────────────────────────
    "i_bVII_bVI_V": {
        "chords": ["i", "-VII", "-VI", "V"],
        "mood": "andalusian cadence, cinematic, falling",
        "valence": -0.5, "arousal": 0.5,
        "typical_scales": ["aeolian", "harmonic_minor", "phrygian"],
    },
    "i_bVI_bVII_i": {
        "chords": ["i", "-VI", "-VII", "i"],
        "mood": "epic minor, metal, powerful",
        "valence": -0.2, "arousal": 0.8,
        "typical_scales": ["aeolian", "pentatonic_minor"],
    },
}


# ============================================================
# Mood presets
# ============================================================

@dataclass
class MoodPreset:
    """
    High-level mood descriptor that bundles scale, chord progression,
    tempo range, and arpeggio pattern guidance into a single named preset.

    All ArpeggioRule fields can be populated from a MoodPreset via
    ``MoodPreset.apply(rule)`` which only overwrites fields that have
    explicit preset values (non-None).
    """
    name: str
    description: str
    scale: str
    progression: str                            # key into CHORD_PROGRESSIONS
    bpm: float                                  # suggested BPM centre
    pattern: List[int]                          # arpeggio degree pattern
    velocity_curve: List[float]
    legato_fraction: float = 0.85
    octave_span: int = 2

    def apply(self, rule: "ArpeggioRule") -> "ArpeggioRule":
        """Return a copy of *rule* with all preset fields applied."""
        from dataclasses import replace
        return replace(
            rule,
            scale=self.scale,
            bpm=self.bpm,
            pattern=list(self.pattern),
            velocity_curve=list(self.velocity_curve),
            legato_fraction=self.legato_fraction,
            octave_span=self.octave_span,
        )


MOOD_PRESETS: Dict[str, MoodPreset] = {
    "serene": MoodPreset(
        name="serene",
        description="Calm, open, floating. Lydian or major pentatonic, slow tempo.",
        scale="pentatonic_major",
        progression="I_IV_V_I",
        bpm=72.0,
        pattern=[0, 1, 2, 3, 2, 1],
        velocity_curve=[0.7, 0.55, 0.65, 0.5, 0.6, 0.5],
        legato_fraction=0.92,
        octave_span=2,
    ),
    "melancholic": MoodPreset(
        name="melancholic",
        description="Sad, introspective, longing. Natural minor, moderate tempo.",
        scale="aeolian",
        progression="i_iv_i_V",
        bpm=84.0,
        pattern=[0, 2, 1, 3, 2, 4, 3, 2],
        velocity_curve=[0.8, 0.6, 0.7, 0.55, 0.65, 0.5, 0.6, 0.45],
        legato_fraction=0.88,
        octave_span=2,
    ),
    "triumphant": MoodPreset(
        name="triumphant",
        description="Bright, confident, resolving. Ionian, strong tempo.",
        scale="ionian",
        progression="I_IV_V_I",
        bpm=120.0,
        pattern=[0, 2, 4, 6, 7, 6, 4, 2],
        velocity_curve=[1.0, 0.75, 0.85, 0.7, 1.0, 0.7, 0.8, 0.65],
        legato_fraction=0.78,
        octave_span=2,
    ),
    "anxious": MoodPreset(
        name="anxious",
        description="Tense, unstable, driven. Phrygian or octatonic, fast tempo.",
        scale="phrygian",
        progression="i_bII_i_bII",
        bpm=148.0,
        pattern=[0, 1, 3, 2, 4, 3, 5, 4],
        velocity_curve=[0.9, 0.8, 1.0, 0.75, 0.95, 0.7, 0.85, 0.65],
        legato_fraction=0.72,
        octave_span=2,
    ),
    "euphoric": MoodPreset(
        name="euphoric",
        description="Bright, floating, expansive. Lydian, upward sweep, fast.",
        scale="lydian",
        progression="I_II_I_II",
        bpm=138.0,
        pattern=[0, 1, 2, 3, 4, 5, 6, 7],
        velocity_curve=[0.85, 0.7, 0.9, 0.75, 1.0, 0.8, 0.9, 0.7],
        legato_fraction=0.80,
        octave_span=2,
    ),
    "mysterious": MoodPreset(
        name="mysterious",
        description="Ambiguous, exotic, searching. Double harmonic or whole-tone.",
        scale="double_harmonic",
        progression="i_VII_VI_V",
        bpm=90.0,
        pattern=[0, 3, 1, 4, 2, 5, 3, 6],
        velocity_curve=[0.7, 0.5, 0.75, 0.55, 0.7, 0.5, 0.65, 0.45],
        legato_fraction=0.86,
        octave_span=2,
    ),
    "bluesy": MoodPreset(
        name="bluesy",
        description="Soulful, gritty, expressive. Blues scale, swung feel.",
        scale="blues",
        progression="blues_12bar",
        bpm=96.0,
        pattern=[0, 2, 1, 3, 2, 4, 3, 5],
        velocity_curve=[1.0, 0.65, 0.85, 0.6, 0.9, 0.55, 0.8, 0.5],
        legato_fraction=0.82,
        octave_span=2,
    ),
    "epic": MoodPreset(
        name="epic",
        description="Dark, powerful, cinematic. Aeolian, wide octave span.",
        scale="aeolian",
        progression="i_bVI_bVII_i",
        bpm=108.0,
        pattern=[0, 4, 2, 6, 4, 8, 6, 10],
        velocity_curve=[1.0, 0.7, 0.85, 0.65, 0.9, 0.6, 0.8, 0.55],
        legato_fraction=0.75,
        octave_span=3,
    ),
    "jazzy": MoodPreset(
        name="jazzy",
        description="Sophisticated, swinging, forward motion. Bebop dominant.",
        scale="bebop_dominant",
        progression="ii_V_I",
        bpm=160.0,
        pattern=[0, 2, 4, 6, 5, 3, 1, 0],
        velocity_curve=[0.9, 0.6, 0.8, 0.55, 0.85, 0.6, 0.75, 0.5],
        legato_fraction=0.70,
        octave_span=2,
    ),
    "folk": MoodPreset(
        name="folk",
        description="Open, earthy, singable. Pentatonic minor, moderate.",
        scale="pentatonic_minor",
        progression="i_iv_i_V",
        bpm=100.0,
        pattern=[0, 1, 2, 3, 4, 3, 2, 1],
        velocity_curve=[0.85, 0.65, 0.75, 0.6, 0.8, 0.6, 0.7, 0.55],
        legato_fraction=0.84,
        octave_span=2,
    ),
}


def semitones_to_hz(root_hz: float, semitones: float) -> float:
    """Equal-temperament conversion: Hz for *semitones* above *root_hz*."""
    return root_hz * (2.0 ** (semitones / 12.0))


def scale_degrees_hz(
    root_hz: float,
    scale: str,
    octave_span: int = 2,
) -> List[float]:
    """Return the Hz values for every scale degree over *octave_span* octaves."""
    if scale not in MODAL_SCALES:
        raise ValueError(
            f"Unknown scale '{scale}'.  Available: {sorted(MODAL_SCALES)}"
        )
    intervals = MODAL_SCALES[scale]
    result: List[float] = []
    for octave in range(octave_span):
        for semitones in intervals:
            result.append(semitones_to_hz(root_hz, semitones + 12 * octave))
    return result


# ============================================================
# Note primitives
# ============================================================

@dataclass
class NoteEvent:
    """One rendered note in the global timeline."""
    fundamental_hz: float
    start_time: float           # seconds, global timeline
    duration_s: float
    velocity: float = 1.0       # 0–1, applied as amplitude scale
    partial_count: int = 6
    # Populated by PhaseHandoff before each render; do not set manually
    # unless you know exactly what you want.
    phase_hints: Dict[int, float] = field(default_factory=dict)
    # harmonic_index → phase_offset in radians at local t=0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_s


@dataclass
class NoteSchedule:
    """Ordered (by start_time) collection of NoteEvents."""
    events: List[NoteEvent] = field(default_factory=list)

    def add(self, event: NoteEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.start_time)

    @property
    def total_duration(self) -> float:
        if not self.events:
            return 0.0
        return max(e.end_time for e in self.events)

    def is_monophonic(self) -> bool:
        """True when no two notes overlap in time."""
        for i in range(1, len(self.events)):
            if self.events[i].start_time < self.events[i - 1].end_time:
                return False
        return True


# ============================================================
# Sequence probability transforms
# ============================================================

import random as _random  # noqa: E402 — kept near its use site


@dataclass
class SequenceProbabilities:
    """
    Stochastic controls applied per-onset in the rhythm schedule.

    All values are probabilities in [0.0, 1.0].

    double_back : Reverse the traversal direction of the progression at this
                  onset (bounces off the bottom — never goes below index 0).
    subversion  : Jump the lookup to the mirror position in the pattern
                  (n_pat−1 − current), acting contrary to momentum.
    chromatic   : Apply a ±1 semitone accidental to the resolved Hz.
    modal       : Apply a ±3 semitone parallel-mode shift to the resolved Hz
                  (same tonal system, alternate modal flavour).
    """
    double_back: float = 0.0
    subversion:  float = 0.0
    chromatic:   float = 0.0
    modal:       float = 0.0

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"double_back", "subversion", "chromatic", "modal"}),
            tracked_inputs=frozenset(),
        )

    def to_dict(self) -> dict:
        return {
            "double_back": self.double_back,
            "subversion":  self.subversion,
            "chromatic":   self.chromatic,
            "modal":       self.modal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceProbabilities":
        sp = cls()
        sp.double_back = float(d.get("double_back", 0.0))
        sp.subversion  = float(d.get("subversion",  0.0))
        sp.chromatic   = float(d.get("chromatic",   0.0))
        sp.modal       = float(d.get("modal",       0.0))
        return sp


class NoteStream:
    """
    Stateful, resettable generator of Hz values from a progression.

    Takes a degree-pattern index list and a Hz degree table; on each
    ``next_hz()`` call it advances through the pattern while applying the
    ``SequenceProbabilities`` transforms.

    Intended use in ``_build_rhythm_schedule``::

        stream = NoteStream(degrees, deg_pattern, patch.seq_probabilities, rng)
        for rep in range(repeats):
            stream.reset()
            for bar_i in ...:
                for step_i in ...:
                    hz = stream.next_hz()
                    if hz is None:
                        continue          # progression exhausted — rest
                    sched.add(NoteEvent(hz, ...))
    """

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        # Pitch generation is driven by note-onset events, not sample clock.
        # All parameters are constant over any block (no live modulation).
        return BlockFaculty(
            constant_inputs=frozenset({"_degrees", "_probs"}),
            tracked_inputs=frozenset(),
        )

    def __init__(
        self,
        degrees:     List[float],
        deg_pattern: List[int],
        probs:       SequenceProbabilities,
        rng:         _random.Random,
    ) -> None:
        self._degrees     = degrees
        self._deg_pattern = deg_pattern
        self._n_degs      = max(1, len(degrees))
        self._n_pat       = max(1, len(deg_pattern))
        self._probs       = probs
        self._rng         = rng
        self._idx         = 0
        self._direction   = 1

    def reset(self) -> None:
        """Reset to the start of the progression (call at the top of each repeat)."""
        self._idx       = 0
        self._direction = 1

    def exhausted(self) -> bool:
        """True when the forward traversal has consumed all notes in the pattern."""
        return self._direction == 1 and self._idx >= self._n_pat

    def next_hz(self) -> float | None:
        """
        Advance and return the next Hz value with transforms applied,
        or ``None`` if the progression is exhausted (rest).
        """
        if self.exhausted():
            return None

        probs = self._probs
        rng   = self._rng

        # ── Double back ─────────────────────────────────────────────────────
        if probs.double_back > 0.0 and rng.random() < probs.double_back:
            self._direction = -self._direction
            if self._direction == -1 and self._idx <= 0:
                self._direction = 1   # can't go below 0; bounce forward

        # ── Effective index (with subversion mirror) ─────────────────────────
        eff_idx = self._idx % self._n_pat
        if probs.subversion > 0.0 and rng.random() < probs.subversion:
            eff_idx = self._n_pat - 1 - eff_idx

        hz = self._degrees[self._deg_pattern[eff_idx] % self._n_degs]

        # ── Chromatic colour: ±1 semitone ────────────────────────────────────
        if probs.chromatic > 0.0 and rng.random() < probs.chromatic:
            hz = hz * (2.0 ** (rng.choice([-1, 1]) / 12.0))

        # ── Modal colour: ±3 semitones ───────────────────────────────────────
        if probs.modal > 0.0 and rng.random() < probs.modal:
            hz = hz * (2.0 ** (rng.choice([-3, 3]) / 12.0))

        # ── Advance index in current direction ───────────────────────────────
        self._idx += self._direction
        if self._idx < 0:
            self._idx       = 0
            self._direction = 1   # bounce off the bottom

        return hz


# ============================================================
# Arpeggiation
# ============================================================

@dataclass
class ArpeggioRule:
    """
    Generates a NoteSchedule from a root pitch, a modal scale, and a rhythm.

    Parameters
    ----------
    root_hz:
        Fundamental of the lowest note in the pattern.
    scale:
        Key into MODAL_SCALES.  Patterns index into the flattened degree list.
    pattern:
        0-based scale-degree indices.  Wraps modulo the available degrees.
    rhythm_beats:
        Duration of each rhythmic step in beats.  Cycles to fill *pattern*.
    bpm:
        Tempo in beats per minute.
    velocity_curve:
        Per-step amplitude 0–1.  Cycles to fill *pattern*.
    legato_fraction:
        Actual note duration = step_duration × legato_fraction.
        1.0 = fully legato (notes touch), <1 = gaps between notes.
    partial_count:
        Harmonic partials per note.
    octave_span:
        How many octaves of scale degrees to build the frequency table from.
    repeats:
        How many times to repeat the full pattern.
    """
    root_hz: float
    scale: str = "pentatonic_minor"
    pattern: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 3, 2, 1])
    rhythm_beats: List[float] = field(default_factory=lambda: [0.5])
    bpm: float = 120.0
    velocity_curve: List[float] = field(default_factory=lambda: [1.0])
    legato_fraction: float = 0.85
    partial_count: int = 6
    octave_span: int = 2
    repeats: int = 1

    @staticmethod
    def block_faculty() -> "BlockFaculty":
        return BlockFaculty(
            constant_inputs=frozenset({"root_hz", "scale", "pattern", "bpm",
                                       "octave_span", "repeats"}),
            tracked_inputs=frozenset(),
        )

    def generate(self) -> NoteSchedule:
        degrees      = scale_degrees_hz(self.root_hz, self.scale, self.octave_span)
        beat_seconds = 60.0 / self.bpm
        schedule     = NoteSchedule()
        t            = 0.0
        n_steps      = len(self.pattern) * self.repeats

        for step in range(n_steps):
            pat_idx  = step % len(self.pattern)
            deg_idx  = self.pattern[pat_idx] % len(degrees)
            hz       = degrees[deg_idx]
            step_dur = self.rhythm_beats[step % len(self.rhythm_beats)] * beat_seconds
            note_dur = max(step_dur * self.legato_fraction, 1.0 / 48000.0)
            velocity = self.velocity_curve[step % len(self.velocity_curve)]

            schedule.add(NoteEvent(
                fundamental_hz=hz,
                start_time=t,
                duration_s=note_dur,
                velocity=velocity,
                partial_count=self.partial_count,
            ))
            t += step_dur

        return schedule


# ============================================================
# Timbre preset
# ============================================================

@dataclass
class TimbrePreset:
    """
    Timbral parameters shared across every note in a rendered sequence.

    amplitude_decay_tau:
        Exponential amplitude envelope τ in seconds.  Shorter = more percussive.
    shape_low / shape_high / shape_center / shape_width:
        Sigmoid shape-state parameters passed to the waveform manifold.
    """
    waveform_manifold: Optional[WaveformManifold] = None
    drift_model: Optional[DriftModel] = None
    amplitude_decay_tau: float = 0.25
    shape_low: float = 0.1
    shape_high: float = 0.7
    shape_center: float = 0.10
    shape_width: float = 0.06
    # Optional post-lattice complex envelope applied per note.
    # Receives note_duration (seconds) and must return a ComplexEnvelope.
    # Use the helpers below — adsr_envelope_factory() is the simplest entry point.
    envelope_factory: Optional[Callable[[float], ComplexEnvelope]] = None

    @classmethod
    def default(cls) -> TimbrePreset:
        """Slightly warped-phase partials with gentle slow drift."""
        return cls(
            waveform_manifold=PhaseWarpedManifold(harmonic_warp_strength=0.20),
            drift_model=SmoothSinusoidalDriftModel(
                base_rate_hz=0.07,
                max_offset_hz=0.03,
                harmonic_scaling_power=1.0,
            ),
        )

    @classmethod
    def pure_sine(cls) -> TimbrePreset:
        """Pure analytic sinusoid, no drift, no warp."""
        return cls(
            waveform_manifold=PureSineManifold(),
            drift_model=NullDriftModel(),
            amplitude_decay_tau=1.0,
            shape_low=0.0,
            shape_high=0.0,
            shape_center=0.0,
            shape_width=1.0,
        )


# ============================================================
# Envelope factories (convenience helpers)
# ============================================================

def adsr_envelope_factory(
    attack_time:   float = 0.005,
    decay_time:    float = 0.04,
    sustain_level: float = 0.75,
    release_time:  float = 0.08,
    peak_level:    float = 1.0,
) -> Callable[[float], ComplexEnvelope]:
    """
    Return a factory that produces an ``AmplitudeEnvelopeShaper(ADSREnvelope(...))``
    configured for each note's duration.

    Assign to ``TimbrePreset.envelope_factory``::

        timbre = TimbrePreset(
            ...,
            envelope_factory=adsr_envelope_factory(
                attack_time=0.008, decay_time=0.05,
                sustain_level=0.70, release_time=0.10,
            ),
        )

    The factory is called once per note with the note's duration in seconds,
    so the release ramp is always placed correctly regardless of note length.
    """
    def _factory(note_duration: float) -> ComplexEnvelope:
        return AmplitudeEnvelopeShaper(
            ADSREnvelope(
                attack_time   = attack_time,
                decay_time    = decay_time,
                sustain_level = sustain_level,
                release_time  = release_time,
                peak_level    = peak_level,
                note_duration = note_duration,
            )
        )
    return _factory


def spline_envelope_factory(
    knots: List[Tuple[float, float]],
    *,
    normalized: bool = True,
) -> Callable[[float], ComplexEnvelope]:
    """
    Return a factory that produces an ``AmplitudeEnvelopeShaper(PiecewiseSplineField(...))``
    for each note.

    *knots* are ``(time, value)`` pairs.  When ``normalized=True`` (default) the
    time coordinates are treated as fractions of the note duration — ``0.0`` is
    note-on and ``1.0`` is note-off — so the same shape applies regardless of
    note length.  Set ``normalized=False`` to supply absolute times in seconds.

    The natural cubic spline passes through every knot and may overshoot between
    them.  For envelopes where overshooting is undesirable (e.g. amplitude must
    stay ≥ 0) prefer :func:`monotone_envelope_factory`.

    Assign to ``TimbrePreset.envelope_factory``::

        timbre = TimbrePreset(
            ...,
            envelope_factory=spline_envelope_factory(
                [(0.0, 0.0), (0.05, 1.0), (0.4, 0.6), (0.9, 0.6), (1.0, 0.0)],
            ),
        )
    """
    def _factory(note_duration: float) -> ComplexEnvelope:
        if normalized:
            scaled = [(t * note_duration, v) for t, v in knots]
        else:
            scaled = list(knots)
        return AmplitudeEnvelopeShaper(PiecewiseSplineField(scaled))
    return _factory


def monotone_envelope_factory(
    knots: List[Tuple[float, float]],
    *,
    normalized: bool = True,
) -> Callable[[float], ComplexEnvelope]:
    """
    Return a factory that produces an ``AmplitudeEnvelopeShaper(PiecewiseMonotoneField(...))``
    for each note.

    *knots* are ``(time, value)`` pairs.  When ``normalized=True`` (default) the
    time coordinates are fractions of the note duration.  Set ``normalized=False``
    for absolute times in seconds.

    PCHIP interpolation is smooth and guaranteed not to overshoot between knots,
    making it the safest choice for amplitude envelopes.

    Assign to ``TimbrePreset.envelope_factory``::

        timbre = TimbrePreset(
            ...,
            envelope_factory=monotone_envelope_factory(
                [(0.0, 0.0), (0.01, 1.0), (0.1, 0.7), (0.85, 0.7), (1.0, 0.0)],
            ),
        )
    """
    def _factory(note_duration: float) -> ComplexEnvelope:
        if normalized:
            scaled = [(t * note_duration, v) for t, v in knots]
        else:
            scaled = list(knots)
        return AmplitudeEnvelopeShaper(PiecewiseMonotoneField(scaled))
    return _factory


def piecewise_envelope_factory(
    knots: List[Tuple[float, float]],
    *,
    normalized: bool = True,
) -> Callable[[float], ComplexEnvelope]:
    """
    Return a factory that produces an ``AmplitudeEnvelopeShaper(PiecewiseLinearField(...))``
    for each note.

    Same ``normalized`` semantics as :func:`spline_envelope_factory`.  Use this
    when you want the exact linear-segment shape with no smoothing.
    """
    def _factory(note_duration: float) -> ComplexEnvelope:
        if normalized:
            scaled = [(t * note_duration, v) for t, v in knots]
        else:
            scaled = list(knots)
        return AmplitudeEnvelopeShaper(PiecewiseLinearField(scaled))
    return _factory


# ============================================================
# Phase handoff
# ============================================================

class PhaseHandoff:
    """
    Computes phase offsets that carry the end-of-note phase into the start
    of the next note, guaranteeing zero discontinuity in every harmonic.

    Monophonic continuity
    ─────────────────────
    At the end of note A (local time *dur_A*), query:

        phases = PhaseHandoff.terminal_phases(driver_A, dur_A)

    Inject into note B:

        note_B.phase_hints = phases

    Then render B on local time [0, dur_B].  Every harmonic partial picks up
    exactly where it left off — no click, no phase jump.

    Harmonic lock for polyphony
    ───────────────────────────
    When notes at different pitches start simultaneously and you want their
    harmonic structures to be constructively related, scale the terminal phases
    by the frequency ratio:

        note_B.phase_hints = PhaseHandoff.harmonic_lock_phases(
            phases, source_hz=f_A, target_hz=f_B
        )

    This places B's fundamental at the same fractional phase cycle as A's,
    scaled by f_B/f_A.
    """

    @staticmethod
    def terminal_phases(
        driver: WitnessAwareSynthDriver,
        local_duration: float,
    ) -> Dict[int, float]:
        """
        Returns ``{harmonic_index: phase_radians}`` for every voice at
        *local_duration* seconds into the note's local render window.
        """
        return {
            v.harmonic_index: v.approximate_effective_phase(local_duration)
            for v in driver.lattice.voices
        }

    @staticmethod
    def harmonic_lock_phases(
        source_phases: Dict[int, float],
        source_hz: float,
        target_hz: float,
    ) -> Dict[int, float]:
        """
        Scale terminal phases of *source_hz* to the frequency space of
        *target_hz*.  The scaling factor is the interval ratio target/source,
        so the i-th harmonic of the new note starts at the same fractional
        cycle position as if it were a continuation of the old note at its
        corresponding harmonic frequency.
        """
        if source_hz == 0.0:
            return dict(source_phases)
        ratio = target_hz / source_hz
        return {h: (phi * ratio) % PI2 for h, phi in source_phases.items()}


# ============================================================
# Lattice builder
# ============================================================

class LatticeBuilder:
    """
    Constructs a ``WitnessAwareSynthDriver`` for a single ``NoteEvent``
    using a ``ConstantPhasePath`` (stable pitch) as the master path.

    All timbral parameters come from the supplied ``TimbrePreset``; per-note
    amplitude is scaled by ``note.velocity / harmonic_index``.

    Rendering settings (witness thresholds, density, projection) are set once
    at construction and reused for every note.
    """

    def __init__(
        self,
        timbre: Optional[TimbrePreset] = None,
        witness_thresholds: Optional[WitnessThresholds] = None,
        density_policy: Optional[DensityPolicy] = None,
        projection_policy: Optional[ProjectionPolicy] = None,
    ) -> None:
        self._timbre     = timbre or TimbrePreset.default()
        self._witness    = witness_thresholds or WitnessThresholds(
            backward_phase_radians=PI2,
            forward_phase_radians=PI2,
            max_support_seconds=3.0,
            support_search_step_seconds=0.001,
            integration_steps_per_check=32,
        )
        self._density    = density_policy or DensityPolicy(
            oversampling_factor=16.0,
            min_sample_rate=48_000.0,
            max_sample_rate=2_000_000.0,
            derivative_weight=0.5,
            support_weight=1.0,
        )
        self._projection = projection_policy or ProjectionPolicy(
            output_sample_rate=48_000.0,
            projection_half_support_seconds=0.002,
            projection_kernel_steps=256,
        )

    def build(self, note: NoteEvent) -> WitnessAwareSynthDriver:
        """Return a fresh driver configured for *note*."""
        t        = self._timbre
        manifold = t.waveform_manifold or PhaseWarpedManifold()
        drift    = t.drift_model or NullDriftModel()

        master_path = ConstantPhasePath(frequency_hz=note.fundamental_hz)

        voices: List[LatticeVoice] = []
        for h in range(1, note.partial_count + 1):
            amp = ExponentialEnvelope(
                initial=note.velocity / h,
                tau_seconds=t.amplitude_decay_tau * (1.0 + 0.05 * h),
            )
            shape = SigmoidField(
                low=t.shape_low,
                high=t.shape_high,
                center=t.shape_center,
                width=t.shape_width,
            )
            voices.append(LatticeVoice(
                name=f"h{h}",
                harmonic_index=h,
                master_phase_path=master_path,
                waveform_manifold=manifold,
                amplitude_field=amp,
                shape_field=shape,
                drift_model=drift,
                gain=1.0,
                phase_offset=note.phase_hints.get(h, 0.0),
            ))

        lattice = HarmonicLattice(voices)

        # Build the per-note complex envelope from the factory if one is set.
        note_envelope: Optional[ComplexEnvelope] = None
        if t.envelope_factory is not None:
            note_envelope = t.envelope_factory(note.duration_s)

        return WitnessAwareSynthDriver(
            lattice=lattice,
            witness_thresholds=self._witness,
            density_policy=self._density,
            projection_policy=self._projection,
            complex_envelope=note_envelope,
        )


# ============================================================
# Sequence renderer
# ============================================================

class SequenceRenderer:
    """
    Renders a full ``NoteSchedule`` into one phase-continuous
    ``AdaptiveSampleBuffer`` spanning the entire global timeline.

    Phase continuity
    ────────────────
    With ``carry_phase=True`` (default), ``PhaseHandoff.terminal_phases`` is
    called after every note and the resulting phases are injected into the
    next note's ``phase_hints``.  For a monophonic arpeggiation this means
    every harmonic partial crosses every note boundary without a phase jump.

    For polyphonic passages the samples from simultaneously active notes are
    merged by superposition: all active notes contribute their
    ``evaluate_linear`` value at each time point in the union of all notes'
    local sample grids, shifted to global time.

    With ``harmonic_lock=True`` the terminal-phase scaling
    (``PhaseHandoff.harmonic_lock_phases``) is applied when the next note has
    a different fundamental, so the two notes share a constructive phase
    relationship at the transition.
    """

    def __init__(
        self,
        builder: Optional[LatticeBuilder] = None,
        harmonic_lock: bool = False,
    ) -> None:
        self._builder       = builder or LatticeBuilder()
        self._harmonic_lock = harmonic_lock

    def render(
        self,
        schedule: NoteSchedule,
        carry_phase: bool = True,
    ) -> AdaptiveSampleBuffer:
        """
        Render every note in *schedule* and return the merged global buffer.

        Parameters
        ----------
        carry_phase:
            Propagate terminal phases from each note to the next.
        """
        if not schedule.events:
            return AdaptiveSampleBuffer()

        local_buffers: List[AdaptiveSampleBuffer] = []
        last_phases: Dict[int, float] = {}
        last_hz: float = 0.0

        for ev in schedule.events:
            if carry_phase and last_phases:
                if self._harmonic_lock and last_hz > 0.0:
                    ev.phase_hints = PhaseHandoff.harmonic_lock_phases(
                        last_phases, last_hz, ev.fundamental_hz
                    )
                else:
                    # Direct copy: every harmonic continues from where it left off.
                    ev.phase_hints = dict(last_phases)

            driver = self._builder.build(ev)
            local_buf = driver.emit_adaptive_complex(EmissionRange(0.0, ev.duration_s))
            local_buffers.append(local_buf)

            if carry_phase:
                last_phases = PhaseHandoff.terminal_phases(driver, ev.duration_s)
                last_hz     = ev.fundamental_hz

        return self._merge(schedule.events, local_buffers)

    @staticmethod
    def _merge(
        events: List[NoteEvent],
        local_buffers: List[AdaptiveSampleBuffer],
    ) -> AdaptiveSampleBuffer:
        """
        Merge time-shifted local buffers into a single sorted global buffer.

        The global time grid is the sorted union of every note's local sample
        times shifted to global coordinates.  At each global time point, all
        currently active notes contribute via ``evaluate_linear`` (linear
        interpolation between their adaptive grid points) and are summed.

        Near-coincident times from different notes' grids (within 1e-10 s) are
        deduplicated to one entry so the strict-increasing contract of
        ``AdaptiveSampleBuffer.add`` is satisfied.
        """
        # Collect (global_t) from every note's adaptive grid
        all_global_times: List[float] = []
        for ev, buf in zip(events, local_buffers):
            for sp in buf.samples:
                all_global_times.append(sp.t + ev.start_time)
        all_global_times.sort()

        # Deduplicate within 1e-10 s
        _EPS = 1e-10
        deduped: List[float] = []
        for t_g in all_global_times:
            if not deduped or t_g - deduped[-1] > _EPS:
                deduped.append(t_g)

        global_buf = AdaptiveSampleBuffer()
        for t_g in deduped:
            value = 0j
            for ev, buf in zip(events, local_buffers):
                local_t = t_g - ev.start_time
                if buf.samples and 0.0 <= local_t <= buf.samples[-1].t:
                    value += buf.evaluate_linear(local_t)
            global_buf.add(t_g, value)

        return global_buf


# ============================================================
# MIDI adapter  (future-proofed stub)
# ============================================================

class MidiAdapter:
    """
    Converts a Standard MIDI File into a ``NoteSchedule``.

    This class is complete for single-track melodic MIDI.  It honours
    ``note_on`` / ``note_off`` messages on all channels and respects
    ``set_tempo`` events.  Program changes, CCs, and SysEx are ignored.

    Requires ``mido``::

        pip install mido

    Usage::

        schedule = MidiAdapter.load("melody.mid", partial_count=6)
        renderer = SequenceRenderer(harmonic_lock=True)
        buf      = renderer.render(schedule)
    """

    _A4_HZ  = 440.0
    _A4_NUM = 69

    @classmethod
    def midi_note_to_hz(cls, note_number: int) -> float:
        """Equal-temperament conversion from MIDI note number to Hz."""
        return cls._A4_HZ * (2.0 ** ((note_number - cls._A4_NUM) / 12.0))

    @classmethod
    def load(
        cls,
        path: str,
        partial_count: int = 6,
        velocity_scale: float = 1.0 / 127.0,
    ) -> NoteSchedule:
        """
        Load *path* (a ``.mid`` file) and return a ``NoteSchedule``.

        Parameters
        ----------
        partial_count:
            Harmonic partials per note.
        velocity_scale:
            Multiplier applied to raw MIDI velocity (0–127) to produce the
            0–1 amplitude velocity stored on each ``NoteEvent``.
        """
        try:
            import mido
        except ImportError as exc:
            raise ImportError(
                "MidiAdapter requires mido — install it with:  pip install mido"
            ) from exc

        mid      = mido.MidiFile(path)
        tempo    = 500_000   # µs/beat = 120 bpm
        ticks_pb = mid.ticks_per_beat

        schedule: NoteSchedule            = NoteSchedule()
        active:   Dict[int, Tuple[float, float]] = {}  # note_num → (start_s, velocity)
        t_s = 0.0

        for msg in mido.merge_tracks(mid.tracks):
            t_s += mido.tick2second(msg.time, ticks_pb, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo

            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (t_s, msg.velocity * velocity_scale)

            elif msg.type in ("note_off", "note_on") and msg.note in active:
                start_s, vel = active.pop(msg.note)
                dur = t_s - start_s
                if dur > 0.0:
                    schedule.add(NoteEvent(
                        fundamental_hz=cls.midi_note_to_hz(msg.note),
                        start_time=start_s,
                        duration_s=dur,
                        velocity=vel,
                        partial_count=partial_count,
                    ))

        return schedule


# ============================================================
# Demo
# ============================================================

def example_render_sequence(output_path: str = "sequence.wav") -> None:
    """
    Render a two-bar pentatonic-minor arpeggiation and write a 16-bit WAV.

    The arpeggio runs over two octaves of A pentatonic minor at 100 bpm,
    stepping through a 8-note falling-then-rising pattern with slight
    velocity variation.  Phase continuity is enforced across every note
    boundary via PhaseHandoff.
    """
    import numpy as np
    from signal_generator_v2 import AdaptiveProjector, ProjectionPolicy

    rule = ArpeggioRule(
        root_hz=110.0,            # A2
        scale="pentatonic_minor",
        pattern=[0, 2, 4, 6, 7, 6, 4, 2,   # one bar up-down
                 1, 3, 5, 7, 8, 7, 5, 3],   # second bar (shifted start)
        rhythm_beats=[0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375,
                      0.375, 0.375, 0.375, 0.375],
        bpm=100.0,
        velocity_curve=[1.0, 0.75, 0.85, 0.7,
                        0.9, 0.7,  0.8,  0.65,
                        1.0, 0.75, 0.85, 0.7,
                        0.9, 0.7,  0.8,  0.65],
        legato_fraction=0.82,
        partial_count=5,
        octave_span=2,
        repeats=1,
    )

    schedule = rule.generate()
    print(f"Schedule: {len(schedule.events)} notes, "
          f"{schedule.total_duration:.2f}s total")

    timbre = TimbrePreset(
        waveform_manifold=PhaseWarpedManifold(harmonic_warp_strength=0.18),
        drift_model=SmoothSinusoidalDriftModel(
            base_rate_hz=0.06,
            max_offset_hz=0.025,
            harmonic_scaling_power=1.0,
        ),
        amplitude_decay_tau=0.22,
        shape_low=0.05,
        shape_high=0.65,
        shape_center=0.08,
        shape_width=0.05,
    )
    projection_policy = ProjectionPolicy(
        output_sample_rate=192_000.0,
        projection_half_support_seconds=0.002,
        projection_kernel_steps=256,
    )

    builder  = LatticeBuilder(
        timbre=timbre,
        projection_policy=projection_policy,
    )
    renderer = SequenceRenderer(builder=builder, harmonic_lock=False)

    print("Rendering notes with phase continuity...")
    global_buf = renderer.render(schedule, carry_phase=True)
    print(f"Global adaptive samples: {len(global_buf)}")

    print("Projecting to uniform grid (numpy)...")
    projector = AdaptiveProjector(projection_policy)
    real_np   = projector.project_real_numpy(
        global_buf, 0.0, schedule.total_duration
    )
    print(f"Uniform output samples: {len(real_np)}")

    import scipy.io.wavfile as _wavfile
    peak  = float(abs(real_np).max()) or 1.0
    f32   = (real_np / peak).astype("float32")
    sr    = int(projection_policy.output_sample_rate)
    _wavfile.write(output_path, sr, f32)
    print(f"Written {output_path}: {len(f32)} samples @ {sr} Hz, 32-bit float mono")


if __name__ == "__main__":
    example_render_sequence()
