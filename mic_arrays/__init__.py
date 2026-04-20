"""
mic_arrays — mic array and receiver preset system.

Usage:
    import mic_arrays
    cfg = mic_arrays.get_array("binaural_standard")
    receivers = cfg.to_cavity_receivers(center_position=(0, 1.4, 2.5), forward_direction=(0, 1, 0))
    baffles   = cfg.to_cavity_baffles(center_position=(0, 1.4, 2.5), forward_direction=(0, 1, 0))

All polar patterns:
    omni, subcardioid, cardioid, supercardioid, hypercardioid, figure8, ribbon, pinnae

Built-in array keys:
    binaural_standard    — two side-facing cardioids + head/torso baffles
    binaural_pinnae      — same + POLAR_PINNAE for interaural module elevation shaping
    xy_stereo            — coincident cardioid pair, ±45°
    ortf                 — 17.5cm / 110° cardioid pair
    ab_pair              — 50cm spaced omni pair
    blumlein             — coincident figure-8 pair
    mid_side             — cardioid mid + figure-8 side
    decca_tree           — 3-omni T arrangement (orchestral standard)
    ribbon_6             — 6 ribbon mics, 50cm spacing
    ribbon_3             — 3 ribbon mics, 50cm spacing
    ambisonic_foa        — tetrahedral FOA A-format
    surround_ring_5      — 5-omni 360° ring
"""
from mic_arrays.types import (
    ALL_POLAR_PATTERNS,
    BaffleSpec,
    MicArrayConfig,
    MicElement,
    POLAR_CARDIOID,
    POLAR_FIGURE8,
    POLAR_HYPERCARDIOID,
    POLAR_OMNI,
    POLAR_PINNAE,
    POLAR_RIBBON,
    POLAR_SUBCARDIOID,
    POLAR_SUPERCARDIOID,
    array_catalog,
    get_array,
    list_arrays,
    pattern_to_directivity_power,
    polar_gain,
    register_array,
)

# Trigger all preset registrations
import mic_arrays.presets  # noqa: F401, E402

__all__ = [
    "ALL_POLAR_PATTERNS",
    "BaffleSpec",
    "MicArrayConfig",
    "MicElement",
    "POLAR_CARDIOID",
    "POLAR_FIGURE8",
    "POLAR_HYPERCARDIOID",
    "POLAR_OMNI",
    "POLAR_PINNAE",
    "POLAR_RIBBON",
    "POLAR_SUBCARDIOID",
    "POLAR_SUPERCARDIOID",
    "array_catalog",
    "get_array",
    "list_arrays",
    "pattern_to_directivity_power",
    "polar_gain",
    "register_array",
]
