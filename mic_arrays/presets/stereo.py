"""
Standard stereo mic array presets.

XY (coincident 90°):
    Two cardioids at the same point, ±45° to Z axis.
    Near-perfect mono compatibility, no phase issues, tight stereo image.

ORTF (17.5cm / 110°):
    Office de Radiodiffusion-Télévision Française standard: two cardioids
    spaced 17.5cm apart, angled ±55° outward.  Excellent stereo image with
    good mono compatibility.

AB spaced pair:
    Two omnis or cardioids spaced far apart (default 50cm).
    Wide stereo, natural, but mono compatibility varies with spacing.

Blumlein pair (coincident figure-8):
    Two ribbon/figure-8 mics at same point, ±45°.
    True stereo with phase-coherent rear pickup, ideal for room ambience.

Mid-Side (MS):
    Mid cardioid facing forward + Side figure-8 facing laterally.
    Variable-width decode: L = (M+S)/2, R = (M-S)/2.
"""
from __future__ import annotations

from mic_arrays.types import (
    MicArrayConfig,
    MicElement,
    POLAR_CARDIOID,
    POLAR_FIGURE8,
    POLAR_OMNI,
    register_array,
)

# ─── XY coincident 90° ───────────────────────────────────────────────────────

XY_STEREO = register_array(MicArrayConfig(
    key="xy_stereo",
    label="XY Stereo (90°)",
    description="Coincident pair of cardioids, ±45° to main axis. Phase-coherent, good mono.",
    array_pattern="xy_stereo",
    output_channels=["left", "right"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="xy_l",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=-45.0,   # rotated 45° left
            self_noise_db=10.0,
            max_spl_db=140.0,
        ),
        MicElement(
            key="xy_r",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=+45.0,   # rotated 45° right
            self_noise_db=10.0,
            max_spl_db=140.0,
        ),
    ],
    baffles=[],
))

# ─── ORTF (17.5cm, 110°) ─────────────────────────────────────────────────────

ORTF = register_array(MicArrayConfig(
    key="ortf",
    label="ORTF (17.5cm / 110°)",
    description=(
        "ORTF standard: two cardioids 17.5cm apart, each angled 55° outward. "
        "Natural stereo image closely matching human hearing geometry."
    ),
    array_pattern="ortf",
    output_channels=["left", "right"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="ortf_l",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(-0.0875, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=-55.0,
            self_noise_db=10.0,
            max_spl_db=140.0,
        ),
        MicElement(
            key="ortf_r",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(+0.0875, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=+55.0,
            self_noise_db=10.0,
            max_spl_db=140.0,
        ),
    ],
    baffles=[],
))

# ─── AB spaced pair (omni) ────────────────────────────────────────────────────

AB_PAIR = register_array(MicArrayConfig(
    key="ab_pair",
    label="AB Spaced Pair (50cm omni)",
    description=(
        "Two omnis 50cm apart. Wide stereo image, natural room pickup. "
        "Mono compatibility depends on source distance."
    ),
    array_pattern="ab_pair",
    output_channels=["left", "right"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="ab_l",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(-0.25, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=8.0,
            max_spl_db=144.0,
        ),
        MicElement(
            key="ab_r",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(+0.25, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=8.0,
            max_spl_db=144.0,
        ),
    ],
    baffles=[],
))

# ─── Blumlein pair (coincident figure-8) ─────────────────────────────────────

BLUMLEIN = register_array(MicArrayConfig(
    key="blumlein",
    label="Blumlein Pair (coincident figure-8)",
    description=(
        "Two ribbon/figure-8 mics at same point, ±45°. "
        "True stereo with phase-coherent rear pickup. "
        "Ideal for ensemble and room ambience capture."
    ),
    array_pattern="blumlein",
    output_channels=["left", "right"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="blumlein_l",
            mic_type="ribbon",
            polar_pattern=POLAR_FIGURE8,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=-45.0,
            self_noise_db=14.0,
            max_spl_db=135.0,
        ),
        MicElement(
            key="blumlein_r",
            mic_type="ribbon",
            polar_pattern=POLAR_FIGURE8,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            rotation_deg=+45.0,
            self_noise_db=14.0,
            max_spl_db=135.0,
        ),
    ],
    baffles=[],
))

# ─── Mid-Side ─────────────────────────────────────────────────────────────────

MID_SIDE = register_array(MicArrayConfig(
    key="mid_side",
    label="Mid-Side (MS)",
    description=(
        "Cardioid mid-mic facing forward + figure-8 side-mic facing laterally. "
        "Decode: L = (M+S)/2, R = (M-S)/2. Variable width via side gain."
    ),
    array_pattern="mid_side",
    output_channels=["mid", "side"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="ms_mid",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=10.0,
            max_spl_db=140.0,
        ),
        MicElement(
            key="ms_side",
            mic_type="ribbon",
            polar_pattern=POLAR_FIGURE8,
            local_position=(0.0, 0.0, 0.0),
            local_direction=(1.0, 0.0, 0.0),   # facing right (+X) as main lobe
            self_noise_db=14.0,
            max_spl_db=135.0,
        ),
    ],
    baffles=[],
))
