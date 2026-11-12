"""Explicit optical correction: separate canonical spectral camera paths.

No new optical law, field estimator, or scattering term is introduced. This is
not the estimator-neutral interval observer. Source without opt-in is unchanged.
"""
VERSION = 'native-field-spectral-continuation-v1'


def separate_field_continuation(source, *, enabled=False):
    if not enabled:
        return source
    if 'int studio_field_band=' in source:
        raise ValueError('separate field continuation already installed')
    first = '        int currentMedium=studio_origin_medium(ro,rd);'
    last = '        for (int i = 0; i < nl; i++) {\n            float measured = lum[i];'
    if source.count(first) != 1 or source.count(last) != 1:
        raise ValueError('canonical sensor continuation extent changed')
    start, end = source.index(first), source.index(last)
    body = source[start:end]
    boundary = 'uFixed3Transport!=0?studio_band_frequency(0):uLayerCentersHz[0]'
    if body.count(boundary) != 1:
        raise ValueError('canonical boundary frequency selection changed')
    body = body.replace(boundary, 'uFixed3Transport!=0?studio_band_frequency(studio_field_band):uLayerCentersHz[studio_field_band]')
    init = 'for (int i = 0; i < 8; i++) throughput[i] = 1.0;'
    if body.count(init) != 1:
        raise ValueError('canonical throughput initialization changed')
    body = body.replace(init, 'for (int i = 0; i < 8; i++) throughput[i] = i==studio_field_band?1.0:0.0;')
    # Every original response/attenuation loop keeps its wavelength index and
    # law; only the active path's slot is evaluated/accumulated. No /band-count
    # weight: each wavelength receives one complete estimator per camera sample.
    body = body.replace('for(int i=0;i<nl;++i)', 'for(int i=studio_field_band;i<studio_field_band+1;++i)')
    body = body.replace('for (int i = 0; i < nl; i++)', 'for (int i = studio_field_band; i < studio_field_band+1; i++)')
    # The camera aperture/jitter ray is shared, but each wavelength reclassifies
    # its own region state. RNG advances sequentially through the existing
    # canonical calls: no reseed, repeated stream, or added random draw.
    prefix = '''        vec3 studio_field_origin=ro,studio_field_direction=rd;
        for(int studio_field_band=0;studio_field_band<(uFixed3Transport!=0?3:nl);++studio_field_band) {
            ro=studio_field_origin;rd=studio_field_direction;
'''
    return source[:start] + prefix + body + '        }\n' + source[end:]
