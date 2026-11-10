"""Explicit interpretation bridge from original colour spectra to bulk optics.

A ColorProfile has no length unit. This conversion is permitted only when its
caller explicitly authors internal-transmittance meaning and a reference length.
It reuses the original spectral evaluator, never baked RGB or an invented curve.
"""
from __future__ import annotations
import numpy as np


def color_profile_transmittance(*, material, profile_name, wavelengths_nm,
                                reference_thickness_mm, interpretation,
                                concentration=1.0, profile_database=None):
    from material_db import EmissionProfileDatabase, ColorProfile, _spd_from_spread, _WL_NM
    if interpretation!='authored_internal_transmittance':
        raise ValueError('colour profile requires explicit internal-transmittance interpretation')
    ep=profile_database if profile_database is not None else EmissionProfileDatabase.instance()
    if profile_name not in ep:raise ValueError('unknown original colour profile')
    profile=ep.get(ep.index_of(profile_name))
    if not isinstance(profile,ColorProfile):raise ValueError('requires a ColorProfile, not emission or remission')
    waves=np.asarray(wavelengths_nm,dtype=np.float64)
    if waves.ndim!=1 or not 2<=len(waves)<=32 or not np.isfinite(waves).all() or np.any(np.diff(waves)<=0) or waves[0]<_WL_NM[0] or waves[-1]>_WL_NM[-1]:
        raise ValueError('wavelengths must increase within original visible spectral grid')
    thickness=float(reference_thickness_mm);strength=float(concentration)
    if not np.isfinite([thickness,strength]).all() or thickness<=0 or strength<0:
        raise ValueError('positive explicit reference thickness and nonnegative concentration required')
    samples=np.interp(waves,_WL_NM,_spd_from_spread(profile.spread.frequency))
    if not np.isfinite(samples).all() or np.any(samples<=0) or np.any(samples>1):
        raise ValueError('original profile is outside internal-transmittance (0,1]; no clamping applied')
    return {'material':material,'wavelengths_nm':waves.tolist(),
            'transmittance':samples.tolist(),'reference_thickness_mm':thickness,
            'concentration':strength}
