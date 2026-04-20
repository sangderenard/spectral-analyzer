"""
Multi-channel and specialty mic array presets.

Decca Tree:
    Three omnis in a T arrangement: one main (forward-center) plus two outriggers.
    Standard orchestral film/classical recording rig. All omnis, spaced for
    spacious hall image.

Ribbon Array (6-mic):
    Six ribbon (figure-8) mics in a horizontal line, evenly spaced.
    Captures the full lateral spread of an ensemble. Each mic's figure-8 pattern
    gives natural room ambience rejection when summed selectively.

Ambisonic First-Order A-Format (FOA):
    Four cardioids at tetrahedral vertices. Encodes complete 3D soundfield.
    Decode to B-format (W, X, Y, Z) for spatial audio.
    Capsule layout: FLU, FRD, BLD, BRU (front-left-up, front-right-down, etc.)
"""
from __future__ import annotations

import math

from mic_arrays.types import (
    MicArrayConfig,
    MicElement,
    POLAR_CARDIOID,
    POLAR_FIGURE8,
    POLAR_OMNI,
    register_array,
)


# ─── Decca Tree ───────────────────────────────────────────────────────────────

DECCA_TREE = register_array(MicArrayConfig(
    key="decca_tree",
    label="Decca Tree",
    description=(
        "Three omnis: main mic 1m forward of center, outriggers 2m apart at center depth. "
        "Standard orchestral recording rig for spacious hall imaging."
    ),
    array_pattern="decca_tree",
    output_channels=["left", "center", "right"],
    interaural_compatible=False,
    elements=[
        MicElement(
            key="decca_c",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(0.0, 0.0, 1.0),    # 1m forward
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=8.0,
            max_spl_db=144.0,
        ),
        MicElement(
            key="decca_l",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(-1.0, 0.0, 0.0),   # 1m left of center
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=8.0,
            max_spl_db=144.0,
        ),
        MicElement(
            key="decca_r",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(+1.0, 0.0, 0.0),   # 1m right of center
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=8.0,
            max_spl_db=144.0,
        ),
    ],
    baffles=[],
))


# ─── 6-mic ribbon array ───────────────────────────────────────────────────────

def _ribbon_array_elements(
    n: int = 6,
    spacing_m: float = 0.50,
    mic_type: str = "ribbon",
) -> list[MicElement]:
    elements = []
    total_width = spacing_m * (n - 1)
    for i in range(n):
        x = -total_width / 2.0 + i * spacing_m
        elements.append(MicElement(
            key=f"ribbon_{i}",
            mic_type=mic_type,
            polar_pattern=POLAR_FIGURE8,
            local_position=(x, 0.0, 0.0),
            local_direction=(0.0, 0.0, 1.0),   # figure-8 null on left/right, lobes front/back
            self_noise_db=14.0,
            max_spl_db=135.0,
        ))
    return elements


RIBBON_6 = register_array(MicArrayConfig(
    key="ribbon_6",
    label="6-Mic Ribbon Array",
    description=(
        "Six ribbon (figure-8) mics in a horizontal line, 50cm spacing. "
        "Captures full lateral ensemble spread. Natural rejection of direct sound "
        "when summed as a figure-8 array. Each mic output is independent."
    ),
    array_pattern="ribbon_6",
    output_channels=[f"ribbon_{i}" for i in range(6)],
    interaural_compatible=False,
    elements=_ribbon_array_elements(n=6, spacing_m=0.50),
    baffles=[],
))

RIBBON_3 = register_array(MicArrayConfig(
    key="ribbon_3",
    label="3-Mic Ribbon Array",
    description="Three ribbon (figure-8) mics, 50cm spacing. Compact version of the 6-mic array.",
    array_pattern="ribbon_3",
    output_channels=["ribbon_0", "ribbon_1", "ribbon_2"],
    interaural_compatible=False,
    elements=_ribbon_array_elements(n=3, spacing_m=0.50),
    baffles=[],
))


# ─── Ambisonic FOA A-format (tetrahedral) ────────────────────────────────────

def _tet_direction(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """Spherical to Cartesian. az=0 → +Z (forward), +az → right, +el → up."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    cos_el = math.cos(el)
    # X=right, Y=up, Z=forward
    x = cos_el * math.sin(az)
    y = math.sin(el)
    z = cos_el * math.cos(az)
    return (x, y, z)


AMBISONIC_FOA = register_array(MicArrayConfig(
    key="ambisonic_foa",
    label="Ambisonic FOA A-Format (tetrahedral)",
    description=(
        "Four cardioids at tetrahedral vertices capturing the full 3D soundfield. "
        "Decode to B-format: W=(FLU+FRD+BLD+BRU)/2, X=(FLU+FRD-BLD-BRU)/2, "
        "Y=(FLU-FRD+BLD-BRU)/2, Z=(FLU-FRD-BLD+BRU)/2."
    ),
    array_pattern="ambisonic_foa",
    output_channels=["FLU", "FRD", "BLD", "BRU"],
    interaural_compatible=False,
    elements=[
        # FLU: azimuth +45° (left-forward), elevation +35.26°
        MicElement(
            key="FLU",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=_tet_direction(-45.0, +35.26),
            self_noise_db=12.0,
            max_spl_db=138.0,
        ),
        # FRD: azimuth -45° (right-forward), elevation -35.26°
        MicElement(
            key="FRD",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=_tet_direction(+45.0, -35.26),
            self_noise_db=12.0,
            max_spl_db=138.0,
        ),
        # BLD: azimuth +135° (left-rear), elevation -35.26°
        MicElement(
            key="BLD",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=_tet_direction(-135.0, -35.26),
            self_noise_db=12.0,
            max_spl_db=138.0,
        ),
        # BRU: azimuth -135° (right-rear), elevation +35.26°
        MicElement(
            key="BRU",
            mic_type="condenser",
            polar_pattern=POLAR_CARDIOID,
            local_position=(0.0, 0.0, 0.0),
            local_direction=_tet_direction(+135.0, +35.26),
            self_noise_db=12.0,
            max_spl_db=138.0,
        ),
    ],
    baffles=[],
))


# ─── Surround spot (5.0 omni ring) ───────────────────────────────────────────

def _ring_elements(n: int, radius_m: float) -> list[MicElement]:
    elements = []
    for i in range(n):
        az = 360.0 * i / n
        az_rad = math.radians(az)
        x = radius_m * math.sin(az_rad)
        z = radius_m * math.cos(az_rad)
        elements.append(MicElement(
            key=f"ring_{i}",
            mic_type="condenser",
            polar_pattern=POLAR_OMNI,
            local_position=(x, 0.0, z),
            local_direction=(0.0, 0.0, 1.0),
            self_noise_db=10.0,
            max_spl_db=140.0,
        ))
    return elements


SURROUND_RING_5 = register_array(MicArrayConfig(
    key="surround_ring_5",
    label="5-Mic Surround Ring (omni)",
    description="Five omnis evenly spaced on a 0.5m radius ring. Captures full 360° room.",
    array_pattern="surround_ring_5",
    output_channels=[f"ring_{i}" for i in range(5)],
    interaural_compatible=False,
    elements=_ring_elements(5, 0.5),
    baffles=[],
))
