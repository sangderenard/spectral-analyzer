"""
room_gravity.py — Gravity model configuration for room regions.

Separate from clip; a region's physical gravity (strength, direction model)
is not a collision concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GravityModel(str, Enum):
    """How gravity behaves inside a region."""
    UNIFORM = "uniform"   # constant downward acceleration
    PANEL   = "panel"     # panel normal defines "down" (sci-fi floor plates)
    NONE    = "none"      # zero-g / free-float


@dataclass
class GravityConfig:
    """
    Gravity parameters for one region.

    Attributes
    ----------
    model
        Which gravity model to apply.
    strength_ms2
        Magnitude of gravitational acceleration in m/s².
    panel_normal
        World-space unit vector pointing away from the gravity source.
        Only used when model == PANEL.
    """
    model:        GravityModel = GravityModel.UNIFORM
    strength_ms2: float        = 9.81
    panel_normal: tuple        = (0.0, 0.0, 1.0)
