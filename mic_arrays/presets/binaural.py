"""
Binaural receiver presets.

binaural_standard:
    Two side-facing cardioid capsules with a rigid head-baffle between them.
    Left ear faces left (-X), right ear faces right (+X).
    The baffle (modeled as the head/body mass) separates the two capsules and
    creates interaural shadowing.  Output should be routed through the interaural
    module (iau_* params) for HRTF elevation notching and fine ITD/ILD shaping.

binaural_pinnae:
    Same geometry as standard but uses the POLAR_PINNAE pattern, which marks the
    capsules for elevation-dependent frequency shaping by the interaural module.
    The pinnae shaping itself is a tie-in point: the interaural module computes
    elevation-dependent spectral coloring based on iau_elevation; this array
    provides the gross spatial layout and the interaural module does the rest.
"""
from __future__ import annotations

from mic_arrays.types import (
    BaffleSpec,
    MicArrayConfig,
    MicElement,
    POLAR_CARDIOID,
    POLAR_PINNAE,
    register_array,
)

# ─── Standard binaural (two side-facing cardioids + head baffle) ─────────────

_HEAD_RADIUS_M = 0.0875   # average adult head radius ~8.75cm → ear spacing 17.5cm
_EAR_HEIGHT_M  = 0.0      # ears at center height of array
_EAR_SPACING_M = _HEAD_RADIUS_M * 2.0

BINAURAL_STANDARD = register_array(MicArrayConfig(
    key="binaural_standard",
    label="Binaural (Head Baffle)",
    description=(
        "Two side-facing cardioid capsules separated by a rigid head baffle. "
        "Left capsule faces -X, right capsule faces +X. "
        "Route output through the interaural module for HRTF shaping."
    ),
    array_pattern="binaural",
    output_channels=["left", "right"],
    interaural_compatible=True,
    elements=[
        MicElement(
            key="ear_l",
            mic_type="pinnae_sim",
            polar_pattern=POLAR_CARDIOID,
            local_position=(-_HEAD_RADIUS_M, _EAR_HEIGHT_M, 0.0),
            local_direction=(-1.0, 0.0, 0.0),  # facing left
            sensitivity_db=0.0,
            self_noise_db=18.0,
            max_spl_db=130.0,
        ),
        MicElement(
            key="ear_r",
            mic_type="pinnae_sim",
            polar_pattern=POLAR_CARDIOID,
            local_position=(+_HEAD_RADIUS_M, _EAR_HEIGHT_M, 0.0),
            local_direction=(+1.0, 0.0, 0.0),  # facing right
            sensitivity_db=0.0,
            self_noise_db=18.0,
            max_spl_db=130.0,
        ),
    ],
    baffles=[
        # Head/body baffle: rigid sphere approximated as flat panel at center,
        # normal pointing right (+X) — separates left from right acoustic space.
        BaffleSpec(
            key="head_baffle",
            local_position=(0.0, 0.0, 0.0),
            local_normal=(1.0, 0.0, 0.0),
            width_m=0.19,
            height_m=0.24,
            reflectivity=0.91,
            normal_reflectivity=0.95,
            grazing_reflectivity=0.74,
            absorption=0.05,
            diffusion=0.08,
            normal_phase_rad=0.10,
            grazing_phase_rad=0.38,
        ),
        # Torso/shoulder baffle: broad panel behind and below, attenuating rear energy
        BaffleSpec(
            key="torso_baffle",
            local_position=(0.0, -0.15, -0.12),
            local_normal=(0.0, 0.5, 1.0),  # tilted slightly upward/forward
            width_m=0.42,
            height_m=0.30,
            reflectivity=0.76,
            normal_reflectivity=0.82,
            grazing_reflectivity=0.58,
            absorption=0.18,
            diffusion=0.28,
            normal_phase_rad=0.0,
            grazing_phase_rad=0.15,
        ),
    ],
))


# ─── Binaural with simulated pinnae pattern ───────────────────────────────────
# Capsules use POLAR_PINNAE to signal to the interaural module that elevation
# shaping should be applied.  Gross directionality is identical to standard.

BINAURAL_PINNAE = register_array(MicArrayConfig(
    key="binaural_pinnae",
    label="Binaural (Simulated Pinnae)",
    description=(
        "Binaural array with simulated pinnae capsules. "
        "Uses POLAR_PINNAE pattern; the interaural module handles elevation-dependent "
        "spectral coloring (iau_elevation → HF notch shaping). "
        "Same head/torso baffle geometry as binaural_standard."
    ),
    array_pattern="binaural",
    output_channels=["left", "right"],
    interaural_compatible=True,
    elements=[
        MicElement(
            key="ear_l",
            mic_type="pinnae_sim",
            polar_pattern=POLAR_PINNAE,
            local_position=(-_HEAD_RADIUS_M, _EAR_HEIGHT_M, 0.01),
            local_direction=(-1.0, 0.0, 0.15),  # slight forward tilt (pinnae geometry)
            sensitivity_db=0.0,
            self_noise_db=18.0,
            max_spl_db=130.0,
        ),
        MicElement(
            key="ear_r",
            mic_type="pinnae_sim",
            polar_pattern=POLAR_PINNAE,
            local_position=(+_HEAD_RADIUS_M, _EAR_HEIGHT_M, 0.01),
            local_direction=(+1.0, 0.0, 0.15),
            sensitivity_db=0.0,
            self_noise_db=18.0,
            max_spl_db=130.0,
        ),
    ],
    baffles=[
        BaffleSpec(
            key="head_baffle",
            local_position=(0.0, 0.0, 0.0),
            local_normal=(1.0, 0.0, 0.0),
            width_m=0.19,
            height_m=0.24,
            reflectivity=0.91,
            normal_reflectivity=0.95,
            grazing_reflectivity=0.74,
            absorption=0.05,
            diffusion=0.08,
            normal_phase_rad=0.10,
            grazing_phase_rad=0.38,
        ),
        BaffleSpec(
            key="torso_baffle",
            local_position=(0.0, -0.15, -0.12),
            local_normal=(0.0, 0.5, 1.0),
            width_m=0.42,
            height_m=0.30,
            reflectivity=0.76,
            normal_reflectivity=0.82,
            grazing_reflectivity=0.58,
            absorption=0.18,
            diffusion=0.28,
            normal_phase_rad=0.0,
            grazing_phase_rad=0.15,
        ),
    ],
))
