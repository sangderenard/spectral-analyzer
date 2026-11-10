"""Region ownership reference for Studio rod/dye/bubble transport integration.

This owns membership, not optics or geometry. Native n/k, Fresnel and existing
parametric/triangle kernels remain the material and intersection authorities.
Not wired into rendering yet. Conflicting rod ownership is deliberately refused.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    kind: str  # glass_rod, injection_dye, air_bubble
    material: int

    def __post_init__(self):
        if self.kind not in ('glass_rod','injection_dye','air_bubble'):
            raise ValueError('unknown Studio region mode')
        if type(self.material) is not int or (self.material != -1 if self.kind=='air_bubble' else self.material<0):
            raise ValueError('invalid region material')


@dataclass(frozen=True)
class ResolvedMedium:
    base_material: int
    # Injection entries represent EXTRA absorption over base host. Different
    # regions of the same dye remain separate concentration contributions.
    dye_materials: tuple[int,...] = ()


def resolve_medium(regions, active, host_material):
    if type(host_material) is not int or host_material<0:
        raise ValueError('invalid host material')
    if not set(active)<=set(regions):raise ValueError('unknown active region')
    occupants=[regions[i] for i in sorted(active)]
    if any(r.kind=='air_bubble' for r in occupants):
        return ResolvedMedium(-1)
    rods={r.material for r in occupants if r.kind=='glass_rod'}
    dyes=tuple(sorted(r.material for r in occupants if r.kind=='injection_dye'))
    if len(rods)>1:
        raise ValueError('different glass rods overlap without authored cut ownership')
    if rods and dyes:
        raise ValueError('rod/dye overlap needs an authored injection target')
    return ResolvedMedium(next(iter(rods)) if rods else host_material,dyes)


@dataclass(frozen=True)
class Crossing:
    region: int
    entering: bool


class RegionState:
    """Per-ray non-LIFO membership; commit only accepted transmission.

    Geometry must supply all coincident crossings as one group. Reflection or
    total internal reflection leaves membership unchanged. Starting inside a
    region requires geometry-derived initial membership, never an air guess.
    """
    def __init__(self,regions,host_material,active=()):
        if any(type(i) is not int or i<0 for i in regions):raise ValueError('invalid region identity')
        self.regions=dict(regions);self.host_material=host_material
        self.active=frozenset(active)
        resolve_medium(self.regions,self.active,self.host_material)

    @property
    def medium(self):
        return resolve_medium(self.regions,self.active,self.host_material)

    def candidate(self,crossings):
        after=set(self.active);seen=set()
        for crossing in crossings:
            if crossing.region in seen or crossing.region not in self.regions or type(crossing.entering) is not bool:
                raise ValueError('invalid or repeated boundary crossing')
            seen.add(crossing.region)
            if crossing.entering:
                if crossing.region in after:raise ValueError('entered an already active volume')
                after.add(crossing.region)
            else:
                if crossing.region not in after:raise ValueError('exited an inactive volume')
                after.remove(crossing.region)
        frozen=frozenset(after)
        return frozen,resolve_medium(self.regions,frozen,self.host_material)

    def cross(self,crossings,*,transmitted):
        if type(transmitted) is not bool:raise ValueError('transmission decision must be explicit')
        after,medium=self.candidate(crossings)
        if transmitted:self.active=after
        return self.medium
