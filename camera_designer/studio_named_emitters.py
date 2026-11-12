"""Derived views of existing named emitter definitions, never another registry."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import yaml
from material_db import EmissionProfile,HistogramSpread,profile_to_rgb

ROOT=Path(__file__).resolve().parents[1]
KEYS=('fluorescent_cool_white','fluorescent_warm_white','led_cool_white','led_warm_white',
      'studio_calibration_nir850')


def named_emitter(key):
    if key not in KEYS:raise ValueError('unknown named Studio emitter profile')
    wavelengths=np.arange(380.,901. if key=='studio_calibration_nir850' else 751.,1.)
    if key.startswith('fluorescent_') or key=='studio_calibration_nir850':
        path=ROOT/'configs'/'lights'/f'{key}.yaml';raw=path.read_bytes();definition=yaml.safe_load(raw)
        # Same normalized Gaussian-mixture law as original LightSpec sampling.
        spec=definition['spectrum']
        if spec.get('type')!='gaussian_peaks':raise ValueError('named fluorescent source law changed')
        density=np.zeros_like(wavelengths)
        for peak in spec['peaks']:
            sigma=float(peak['fwhm_nm'])/(2*math.sqrt(2*math.log(2)))
            density+=float(peak['weight'])/sum(float(x['weight']) for x in spec['peaks'])*np.exp(-.5*((wavelengths-float(peak['nm']))/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        label=definition.get('description',key)
    else:
        from .emitter_profile import emitter_from_dict
        path=ROOT/'presets'/'emitters'/f'{key}.emitter.yaml';raw=path.read_bytes();definition=yaml.safe_load(raw)
        profile=emitter_from_dict(definition)
        # Calls the original composite evaluator, including weighted components.
        density=np.array([profile.power_at(w/1000.,1.) for w in wavelengths],float)
        label=profile.label or profile.name
    if not np.isfinite(density).all() or np.any(density<0) or not density.sum()>0:raise ValueError('invalid named emitter spectrum')
    density=density/np.trapezoid(density,wavelengths)
    # Original source sampling changes wavelength, not packet energy. The
    # profile RGB API expects dimensionless SPD relative to equal-energy white,
    # so convert PDF [1/nm] to relative SPD; this is not peak normalization.
    native=EmissionProfile(spd=HistogramSpread(wavelengths,density*(750.-380.)),total_power_W=1.)
    rgb=np.asarray(profile_to_rgb(native),float)
    preview=np.maximum(rgb,0);preview/=max(float(preview.max()),1e-30)
    from .studio_spectral_reflector import authored_reflector_bands
    columns=[]
    for channel in range(3):
        row=np.zeros(16);row[channel]=1
        response=np.array([b['reflectance'] for b in authored_reflector_bands(row,wavelengths/1000.)])
        filtered=EmissionProfile(spd=HistogramSpread(wavelengths,density*370.*response),total_power_W=1.)
        columns.append(profile_to_rgb(filtered,clamp_negative=False))
    matrix=np.column_stack(columns)
    return dict(key=key,label=label,source=path.relative_to(ROOT).as_posix(),revision=hashlib.sha256(raw).hexdigest(),definition=definition,
                wavelengths_nm=wavelengths.tolist(),spectral_power=density.tolist(),preview_rgb=np.maximum(rgb,0).tolist(),preview_chromaticity=preview.tolist(),filter_rgb_matrix=matrix.tolist(),
                projection='original named spectral shape distributes existing fixture gain; PDF converted relative to equal-energy white for native RGB; fixed1/3 integrated-bin approximation; no peak normalization or calibrated-watt claim'),native


def export_named_emitters():
    return {'version':1,'authority':'existing Spectral LightSpec/EmitterProfile definitions',
            'records':[named_emitter(key)[0] for key in KEYS]}


def transport_band_power(record,wavelengths_um,filter_rgb=None):
    """Distribute existing fixed-band source power using original spectral PDF.

    Full shape stays in the named source. Fixed1/3 bins integrate its probability
    mass; sum equals band count, preserving the pre-existing unit-white fixture
    power budget per scalar gain. No PDF-height dimming or peak normalization.
    """
    grid=np.asarray(wavelengths_um,float)*1000
    if len(grid) not in (1,3,4,8) or len(set(grid))!=len(grid):raise ValueError('named source supports fixed1/fixed3/fixed4/fixed8 transport')
    order=np.argsort(grid);ordered=grid[order];edges=np.r_[350. if len(grid)==8 else 380.,(ordered[:-1]+ordered[1:])*.5,1000. if len(grid)==8 else 900. if len(grid)==4 else 750.]
    w=np.asarray(record['wavelengths_nm']);d=np.asarray(record['spectral_power']);energy=[]
    if filter_rgb is not None:
        from .studio_spectral_reflector import authored_reflector_bands
        row=np.zeros(16);row[:3]=filter_rgb
        d=d*np.array([b['reflectance'] for b in authored_reflector_bands(row,w/1000.)])
    for lo,hi in zip(edges[:-1],edges[1:]):
        x=np.r_[lo,w[(w>lo)&(w<hi)],hi]
        energy.append(np.trapezoid(np.interp(x,w,d,left=0.,right=0.),x))
    result=np.empty(len(grid));result[order]=energy;result*=len(grid)
    return result

def validate_named_emitter_refs(value,material_count):
    if value is None:return {}
    if not isinstance(value,dict) or set(value)!={'version','records'} or type(value['version']) is not int or value['version']!=1 or not isinstance(value['records'],list):raise ValueError('invalid named emitter references')
    if len(value['records'])>material_count:raise ValueError('too many emitter references')
    result={}
    for reference in value['records']:
        if not isinstance(reference,dict) or set(reference)!={'material','profile','revision','gain','filter'}:raise ValueError('invalid named emitter record')
        material=reference['material']
        if type(material) is not int or not 0<=material<material_count or material in result:raise ValueError('invalid or repeated emitter material')
        definition,_=named_emitter(reference['profile'])
        if reference['revision']!=definition['revision']:raise ValueError('named emitter revision changed; refresh authored profile before capture')
        gain=reference['gain']
        if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not 0<=gain<=1e6:raise ValueError('invalid named emitter gain/filter')
        try:f=None if reference['filter'] is None else np.asarray(reference['filter'],float)
        except (OverflowError,TypeError,ValueError):raise ValueError('invalid named emitter gain/filter') from None
        if not math.isfinite(gain) or (f is not None and (f.shape!=(3,) or not np.isfinite(f).all() or np.any((f<0)|(f>1)))):raise ValueError('invalid named emitter gain/filter')
        result[material]=dict(reference,filter=None if f is None else f.tolist(),gain=float(gain))
    return result


def named_source_material(reference,wavelengths_um,profile_database,surface=None):
    """Resolve original named source into shared profile/material native buffers."""
    from .studio_spectral_reflector import authored_reflector_bands
    definition,native=named_emitter(reference['profile'])
    if reference['revision']!=definition['revision']:raise ValueError('named emitter source revision mismatch')
    gain=float(reference['gain']);filter_rgb=None if reference['filter'] is None else np.asarray(reference['filter'],float)
    power=transport_band_power(definition,wavelengths_um,filter_rgb)*gain
    fingerprint=hashlib.sha256(json.dumps(reference,sort_keys=True).encode()+np.asarray(wavelengths_um,np.float64).tobytes()).hexdigest()
    key='studio_named_emitter:'+fingerprint
    # Preserve the original native profile radiance; dimmer is a scalar gain.
    # No new peak/white/energy normalization is inserted here.
    if filter_rgb is None:
        full_filter=np.ones(len(definition['wavelengths_nm']))
    else:
        filter_row=np.zeros(16);filter_row[:3]=filter_rgb
        full_filter=np.array([b['reflectance'] for b in authored_reflector_bands(filter_row,np.asarray(definition['wavelengths_nm'])/1000.)])
    native.spd=HistogramSpread(np.asarray(definition['wavelengths_nm']),np.asarray(definition['spectral_power'])*(750.-380.)*full_filter)
    native.total_power_W=gain
    preview=np.asarray(profile_to_rgb(native),float)
    if key not in profile_database:profile_database.register(key,native)
    from material_db import MaterialDatabase
    base=dict(surface or dict(albedo=[1,1,1],roughness=1,metallic=0,transmission=0,ior=1,opacity=1))
    projection=MaterialDatabase();projection.register('surface',base)
    surface_bands=projection.build_mat_buf(freq_hz=299792458./(np.asarray(wavelengths_um)*1e-6))[:len(wavelengths_um)]
    bands=[]
    for i,(wavelength,energy) in enumerate(zip(wavelengths_um,power)):
        frequency=299792458./(wavelength*1e-6)
        bands.append(dict(center_hz=frequency,bandwidth_hz=frequency*1e-4,reflectance=float(surface_bands[i,2]),transmittance=float(surface_bands[i,3]),diffuse_frac=float(surface_bands[i,4]),emission=float(energy),reemission=float(surface_bands[i,6]),ior_real=float(surface_bands[i,7]),ior_imag=float(surface_bands[i,8])))
    return key,dict(base,emission_rgb=preview.tolist(),emit_profile_idx=float(profile_database.index_of(key)),spectral_bands=bands)
