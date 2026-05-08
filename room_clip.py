"""
room_clip.py — Clip configuration for the room region hierarchy.

Defines the DATA side of how rooms constrain player movement.
No solver logic lives here; these objects are consumed by whatever collision
backend is active (PlayerClipEngine, pure-Python slab, etc.) and by the Room
class to describe a deployed region's physical rules.

Gravity is a separate concern — see room_gravity.py.

Hierarchy
---------
    RegionAuthority forms a parent chain.  A sub-region inherits from its
    parent but can override any setting.  Full spatial membership queries
    (object_in_region, player_in_region) are stubbed — implementation is
    deferred pending the broader entity-graph integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import Dict, FrozenSet, Optional, Set


# ── Clip surface flags ───────────────────────────────────────────────────────

class ClipSurface(Flag):
    """Which room surfaces participate in player clip resolution."""
    NONE     = 0
    FLOOR    = auto()
    WALLS    = auto()
    CEILING  = auto()
    STANDARD = FLOOR | WALLS        # typical first-person locomotion
    ALL      = FLOOR | WALLS | CEILING


# ── Clip configuration ───────────────────────────────────────────────────────

@dataclass
class ClipConfig:
    """
    Full description of how a region clips entities.

    Attributes
    ----------
    clip_surfaces
        Which room boundary surfaces act as clip geometry.
    objects_clip
        Whether placed objects inside the region also block movement.
    player_radius_m
        Collision sphere radius for the player.
    clip_authority
        Which backend resolves collisions.  One of:
          "engine"  — C++ PlayerClipEngine (preferred when compiled)
          "python"  — pure-Python axis-aligned slab solver
          "none"    — clipping disabled for this region
    """
    clip_surfaces:   ClipSurface = ClipSurface.STANDARD
    objects_clip:    bool        = False
    player_radius_m: float       = 0.25
    clip_authority:  str         = "engine"


# ── Region authority ─────────────────────────────────────────────────────────

class RegionAuthority:
    """
    Named authority over a spatial region.

    Registers in a process-global dict keyed by region_id so any system can
    look up a region by name.  The parent chain supports hierarchical
    authority: a sub-room overrides only what it needs and inherits the rest.

    Entity membership (register_entity / unregister_entity) is provided as a
    stub — the full spatial containment test is deferred.

    Usage
    -----
        auth = RegionAuthority.for_room("studio_a")
        auth.clip_config.clip_surfaces = ClipSurface.ALL
    """

    _registry: Dict[str, "RegionAuthority"] = {}

    def __init__(
        self,
        region_id: str,
        parent: Optional["RegionAuthority"] = None,
    ) -> None:
        self.region_id:   str                       = region_id
        self.parent:      Optional[RegionAuthority] = parent
        self.clip_config: ClipConfig                = ClipConfig()
        self._entity_ids: Set[str]                  = set()
        RegionAuthority._registry[region_id] = self

    # ── Registry ─────────────────────────────────────────────────────────────

    @classmethod
    def get(cls, region_id: str) -> Optional["RegionAuthority"]:
        """Look up a region by ID.  Returns None if not found."""
        return cls._registry.get(region_id)

    @classmethod
    def all_regions(cls) -> Dict[str, "RegionAuthority"]:
        return dict(cls._registry)

    # ── Authority chain ───────────────────────────────────────────────────────

    def effective_clip_config(self) -> ClipConfig:
        """
        Return the effective ClipConfig for this region.

        Currently returns self.clip_config directly.  Future: walk the parent
        chain and merge, letting child settings override parent defaults.
        """
        return self.clip_config

    # ── Entity membership stubs ───────────────────────────────────────────────

    def register_entity(self, entity_id: str) -> None:
        """Record that an entity has entered this region."""
        self._entity_ids.add(entity_id)

    def unregister_entity(self, entity_id: str) -> None:
        self._entity_ids.discard(entity_id)

    def object_has_clip_authority(self, obj_id: str) -> bool:
        """True when placed objects in this region act as clip geometry."""
        return self.clip_config.objects_clip

    @property
    def entity_ids(self) -> FrozenSet[str]:
        return frozenset(self._entity_ids)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_room(
        cls,
        room_id: str,
        *,
        clip_surfaces:  ClipSurface = ClipSurface.STANDARD,
        objects_clip:   bool        = False,
        clip_authority: str         = "engine",
        parent:         Optional["RegionAuthority"] = None,
    ) -> "RegionAuthority":
        """Create a RegionAuthority pre-configured for a standard room."""
        auth = cls(room_id, parent=parent)
        auth.clip_config = ClipConfig(
            clip_surfaces   = clip_surfaces,
            objects_clip    = objects_clip,
            clip_authority  = clip_authority,
        )
        return auth

    def __repr__(self) -> str:
        return (f"RegionAuthority({self.region_id!r}, "
                f"surfaces={self.clip_config.clip_surfaces!r})")
