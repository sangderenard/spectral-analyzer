"""Bounded normal-ray test of curve-shaped, index-matched absorbing layers.

This is a physical absorption probe, not a mixed-scene reader registration.
It uses the exact cookie contours and the existing shared optical material.
There is no bitmap, display opacity, lens replacement or post-exposure tint.
"""
from __future__ import annotations

import math
import numpy as np

from .curve_cookie import CurveCookie
from .optical_material import OpticalMaterial
from .studio_optical_media import optical_extinction, validate_optical_media


class CurveFilterProbe:
    """A finite, normal-incidence absorption-only test through a known path.

    The contour's open region has ``text_density``; the rest of the finite
    rectangle has ``background_density``. Both multiply the shared material's
    concentration, not RGB opacity. An optional actually compiled density
    product can replace this piecewise-constant field in subsequent work.
    """

    def __init__(self, cookie, record, pbr, *, thickness_m, text_density=0., background_density=1.):
        if not isinstance(cookie, CurveCookie):
            raise TypeError('Expected exact CurveCookie geometry')
        self.cookie = CurveCookie.from_dict(cookie.to_dict())
        material = record.get('material')
        self.record = validate_optical_media({'version': 1, 'materials': [record]}, np.asarray(pbr), [material])[material]
        self.thickness_m = float(thickness_m)
        self.text_density = float(text_density)
        self.background_density = float(background_density)
        if not math.isfinite(self.thickness_m) or self.thickness_m <= 0:
            raise ValueError('Filter material path must be positive metres')
        if not all(math.isfinite(x) and x >= 0 for x in (self.text_density,self.background_density)):
            raise ValueError('Relative density must be nonnegative')

    def trace_normal(self, origin, direction, wavelength_um):
        """Trace one normal ray; return actual internal power transmission.

        Arbitrary-angle paths, refracting interfaces and diffraction are not
        implemented here and are never silently replaced by entry-point tint.
        """
        ro, rd = np.asarray(origin, np.float64), np.asarray(direction, np.float64)
        if ro.shape != (3,) or rd.shape != (3,) or not np.isfinite([ro,rd]).all() or rd[2] == 0:
            raise ValueError('Expected finite nonzero normal ray')
        if rd[0] != 0 or rd[1] != 0:
            raise ValueError('This probe requires normal incidence and an index-matched path')
        # Validate actual wavelength even when the ray misses the filter.
        k = optical_extinction(self.record, wavelength_um)
        t = (self.cookie.z_pos-ro[2])/rd[2]
        h = ro+t*rd
        intersects = t > 1e-9 and abs(h[0]) <= self.cookie.half_size[0] and abs(h[1]) <= self.cookie.half_size[1]
        if not intersects:
            return {'intersects':False,'transmittance':1.,'path_m':0.,'relative_density':0.}
        density = self.text_density if self.cookie.open_at(*h[:2]) else self.background_density
        medium = OpticalMaterial(name='shared_curve_filter_probe', n_d=1., k=k)
        value = medium.transmittance(self.thickness_m*density, wavelength_um)
        return {'intersects':True,'transmittance':value,'path_m':self.thickness_m,
                'relative_density':density,'wavelength_um':float(wavelength_um),
                'shared_extinction_k':k,'ray_parameter':float(t)}

