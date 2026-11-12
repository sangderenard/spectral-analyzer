"""Versioned internal transmittance at known thickness, projected to native n/k.

The record is an authored spectral approximation, not measured data by default.
Interface Fresnel loss is excluded. No alpha, roughness or RGB conversion enters
bulk extinction. Distances are metres; input wavelengths are nanometres.
"""
from __future__ import annotations
import math
import numpy as np


def validate_optical_media(value, pbr, required_ids):
    if not isinstance(value, dict) or set(value) != {'version', 'materials'} or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('expected version 1 optical media')
    rows = value['materials']
    if not isinstance(rows, list) or not rows or len(rows) > len(pbr):
        raise ValueError('invalid optical material count')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'material', 'wavelengths_nm', 'transmittance', 'reference_thickness_mm', 'concentration'}:
            raise ValueError('invalid reference-thickness optical material')
        material = row['material']
        if type(material) is not int or not 0 <= material < len(pbr) or material in result:
            raise ValueError('invalid or repeated optical material ID')
        for key in ('reference_thickness_mm', 'concentration'):
            x = row[key]
            if type(x) not in (float, int) or not math.isfinite(x):
                raise ValueError('optical thickness/concentration must be finite numbers')
        if row['reference_thickness_mm'] <= 0 or row['concentration'] < 0:
            raise ValueError('optical thickness must be positive and concentration nonnegative')
        w, t = row['wavelengths_nm'], row['transmittance']
        if not isinstance(w, list) or not isinstance(t, list) or not 2 <= len(w) <= 32 or len(w) != len(t):
            raise ValueError('optical spectrum requires 2 to 32 matching samples')
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in w+t):
            raise ValueError('optical spectrum must be finite numbers')
        if w[0] <= 0 or any(b <= a for a,b in zip(w,w[1:])) or any(not 0 < x <= 1 for x in t):
            raise ValueError('wavelengths must increase; internal transmittance must be in (0,1]')
        distance_m = row['reference_thickness_mm'] * .001
        if distance_m == 0:
            raise ValueError('reference thickness underflows metre units')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            sigma = -np.log(np.asarray(t, np.float64)) / distance_m * row['concentration']
            max_k = sigma * np.asarray(w) * 1e-9 / (4*math.pi)
        if not np.isfinite(sigma).all() or not np.isfinite(max_k).all() or np.any(max_k > np.finfo(np.float32).max):
            raise ValueError('optical extinction exceeds the native float representation')
        # The shared Studio boundary kernel currently uses real-index
        # dielectric Fresnel, not complex-index interface reflection. Admit
        # only weak absorption (k <= 1e-3); stronger media require that kernel
        # before their bulk transmittance can be advertised as supported.
        # This is a declared approximation regime, not a universal material law.
        peak_k=float(np.max(max_k))
        for j in range(len(w)-1):
            slope=(sigma[j+1]-sigma[j])/(w[j+1]-w[j])
            intercept=sigma[j]-slope*w[j]
            if slope<0:
                peak=-intercept/(2*slope)
                if w[j]<peak<w[j+1]:
                    peak_k=max(peak_k,float((slope*peak+intercept)*peak*1e-9/(4*math.pi)))
        if peak_k > 1e-3:
            raise ValueError('strong optical extinction requires complex-index interface transport (k > 0.001)')
        p = pbr[material]
        if p[4] != 0 or p[5] <= 0 or p[7] != 0 or p[6] < 1 or np.any(p[8:11] != 0):
            raise ValueError('optical media require nonmetal, nonemissive dielectric PBR with opacity zero')
        result[material] = {key: (list(val) if isinstance(val,list) else val) for key,val in row.items()}
    if set(result) != set(required_ids):
        raise ValueError('optical definitions must cover exactly the explicit volume materials')
    return result


def optical_extinction(record, wavelength_um, *, outside_policy="error"):
    """Interpolate authored absorption; explicitly clear unmeasured game bands."""
    nm = float(wavelength_um)*1000
    w = record['wavelengths_nm']
    if not math.isfinite(nm):
        raise ValueError('transport wavelength outside authored optical spectrum')
    if not w[0] <= nm <= w[-1]:
        if outside_policy == "game-transparent-outside-authored-v1":
            return 0.0
        raise ValueError('transport wavelength outside authored optical spectrum')
    sigma = -np.log(np.asarray(record['transmittance'], np.float64)) / (record['reference_thickness_mm']*.001)
    sigma *= record['concentration']
    return float(np.interp(nm,w,sigma)) * nm*1e-9/(4*math.pi)


def validate_region_absorption_bound(regions, optics):
    """Conservative upper bound for additive dye mixtures under real-n Fresnel.

    Sum per-spectrum maxima, counting each dye volume (including repeated
    material IDs). This can reject nonoverlapping strong dyes conservatively;
    it must not admit a strong mixture merely because components are weak.
    """
    def peak(record):
        w=np.asarray(record['wavelengths_nm'],np.float64)
        sigma=-np.log(np.asarray(record['transmittance'],np.float64))/(record['reference_thickness_mm']*.001)*record['concentration']
        values=list(sigma*w*1e-9/(4*math.pi))
        for j in range(len(w)-1):
            slope=(sigma[j+1]-sigma[j])/(w[j+1]-w[j])
            intercept=sigma[j]-slope*w[j]
            if slope<0:
                x=-intercept/(2*slope)
                if w[j]<x<w[j+1]:values.append((slope*x+intercept)*x*1e-9/(4*math.pi))
        return max(values)
    host=peak(optics[regions[0]['material']])
    dyes=sum(peak(optics[r['material']]) for r in regions if r['mode']=='injection_dye')
    if host+dyes>1e-3:
        raise ValueError('combined dye extinction requires complex-index interface transport (k bound > 0.001)')
