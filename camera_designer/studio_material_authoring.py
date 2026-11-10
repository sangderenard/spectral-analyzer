"""Fail-closed canonical source adapter for the saved Studio scene ABI."""
from canonical_material_documents import restore_material_document, native_payload, native_volume_payload


SCENE_MATERIAL_CAPABILITIES = {
    "ray_coefficients": "native",
    "spectral_bands": "native transport-grid projection",
    "pbr": "native derived surface view",
    "enamel": "preserved in source/native database; not bound by Studio scene",
    "texture_stack": "preserved in source/native database; not bound by Studio scene",
    "color_profile": "not consumed by Studio surface shader",
    "emission_profile": "requires spectral source adapter; RGB substitution forbidden",
    "notes": "metadata only",
    "gameplay": "metadata only",
}


def compile_scene_authoring(authoring, material_count, profile_database, *, native_volume=False):
    if not isinstance(authoring, dict) or set(authoring) != {'version', 'records'} or type(authoring['version']) is not int or authoring['version'] != 1:
        raise ValueError('material authoring requires version 1 and records')
    if not isinstance(authoring['records'], list):
        raise ValueError('material authoring records must be a list')
    result = {}
    for record in authoring['records']:
        if not isinstance(record, dict) or set(record) != {'material', 'source', 'overrides'}:
            raise ValueError('invalid canonical material authoring record')
        index = record['material']
        if type(index) is not int or not 0 <= index < material_count or index in result:
            raise ValueError('canonical material ID invalid or duplicated')
        doc = restore_material_document(record['source'], record['overrides'])
        adapter = native_volume_payload if native_volume else native_payload
        compiled = adapter(doc, profile_database=profile_database)
        payload = compiled.payload
        unsupported = list(compiled.uncompiled_fields)
        if payload.get("reactive_shift_hz", 0) != 0:
            unsupported.append("reactive_shift_hz (Studio shader adapter unavailable)")
        if payload.get("mat_flags", 0) != 0:
            unsupported.append("mat_flags (Studio geometry derives its own flags)")
        # These have native database records but are not transported by this
        # scene tuple / sensor shader. Never imply preservation means rendering.
        for field in ('enamel', 'texture_stack'):
            if payload.get(field): unsupported.append(field)
        for field in ('color_profile_idx', 'emit_profile_idx', 'remit_profile_idx'):
            if float(payload.get(field, -1)) >= 0: unsupported.append(field)
        nested = payload.get('pbr', {})
        emission = nested.get('emission_rgb', nested.get('emission', payload.get('emission_rgb', [0, 0, 0])))
        if any(float(v) != 0 for v in emission): unsupported.append('emission spectrum')
        for band in payload.get('bands', payload.get('spectral_bands', [])):
            if any(float(band.get(k, 0)) != 0 for k in ('emission', 'reemission')):
                unsupported.append('spectral emission/reemission')
        if unsupported:
            raise ValueError(f'canonical material {doc.key}: Studio adapter does not transport {", ".join(sorted(set(unsupported)))}')
        result[index] = compiled
    return result


def validate_canonical_volume_optics(marker, canonical, required_ids, regions=None, wavelengths_um=(.5875618,)):
    """Explicitly choose authored native n/k, not reference-T or display colour."""
    import numpy as np
    from material_db import MaterialDatabase
    if marker != {'version': 1, 'source': 'native_spectral_bands'} or type(marker.get('version')) is not int:
        raise ValueError('canonical volume optics requires explicit native_spectral_bands version 1')
    if set(canonical) != set(map(int, required_ids)):
        raise ValueError('canonical n/k sources must cover exactly all volume materials')
    peaks={}
    for material,compiled in canonical.items():
        payload=compiled.payload
        bands=payload.get('bands',payload.get('spectral_bands',[]))
        if not bands:
            raise ValueError('canonical volume optics requires explicit native spectral n/k bands')
        for band in bands:
            if 'ior_real' not in band or 'ior_imag' not in band:
                raise ValueError('canonical volume bands require explicit ior_real and ior_imag')
            n,k=float(band['ior_real']),float(band['ior_imag'])
            if not np.isfinite([n,k]).all() or n<1 or not 0<=k<=.001:
                raise ValueError('canonical volume requires n >= 1 and weak absorption 0 <= k <= .001')
        isolated=MaterialDatabase();isolated.register('volume',payload)
        row=isolated.build_tensors()['pbr'][0]
        if row[4]!=0 or row[6]<1 or np.any(row[8:11]!=0):
            raise ValueError('canonical volume requires nonmetal nonemissive dielectric')
        grid=299792458.0/(np.asarray(wavelengths_um,dtype=np.float64)*1e-6)
        projected=isolated.build_mat_buf(freq_hz=grid)[:len(grid)]
        if not np.isfinite(projected).all() or np.any(projected[:,7]<1) or np.any(projected[:,8]<0) or np.any(projected[:,8]>.001):
            raise ValueError('native transport-grid projection violates dielectric n/k limits')
        peaks[material]=max(float(b['ior_imag']) for b in bands)
    if regions:
        bound=peaks[regions[0]['material']]+sum(peaks[r['material']] for r in regions if r['mode']=='injection_dye')
        if bound>.001:
            raise ValueError('combined canonical dye extinction exceeds weak-absorption k bound .001')
    return peaks
