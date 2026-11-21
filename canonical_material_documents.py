"""Lossless source-document boundary for the canonical material YAML library.

No function writes a file. Unedited bytes and revision are exact. Editing an
existing field returns a new YAML document preserving all other parsed fields;
PyYAML does not promise comment/format preservation after an edit. Native tensor
projection is explicit and downstream-only, never a replacement authoring source.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct
from typing import Any
import yaml

MATERIAL_DIRECTORY=Path(__file__).resolve().parent/'configs'/'materials'
_KEY=re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\Z')


def _key(value):
    if not isinstance(value,str) or not _KEY.fullmatch(value):
        raise ValueError('material key must be a plain library filename stem')
    return value


@dataclass(frozen=True)
class MaterialDocument:
    key: str
    source_path: str
    source_bytes: bytes

    def __post_init__(self):
        _key(self.key)
        if not isinstance(self.source_bytes,bytes):
            raise TypeError('material source must be bytes')
        self.definition  # Validate without retaining a caller-mutable mapping.

    @property
    def revision(self):
        return hashlib.sha256(self.source_bytes).hexdigest()

    @property
    def definition(self):
        value=yaml.safe_load(self.source_bytes.decode('utf-8'))
        if not isinstance(value,dict):
            raise ValueError('material document root must be a mapping')
        result = _independent_tree(value)
        if 'medium' in result:
            from material_medium import validate_medium_profile
            result['medium'] = validate_medium_profile(result['medium'])
        return result

    def with_field(self,path,value):
        """Replace one existing mapping/list field; never create hidden structure."""
        if not isinstance(path,(tuple,list)) or not path:
            raise ValueError('field path must be a nonempty key/index sequence')
        root=self.definition;node=root
        for part in path[:-1]:
            node=_child(node,part)
        old=_child(node,path[-1])
        if old==value and type(old) is type(value):
            return self
        # Deep-copy prevents edits to the caller's value changing this document.
        node[path[-1]]=copy.deepcopy(value)
        raw=yaml.safe_dump(root,sort_keys=False,allow_unicode=True).encode('utf-8')
        return MaterialDocument(self.key,self.source_path,raw)

    def snapshot(self):
        return {'key':self.key,'revision':self.revision,
                'source_yaml':self.source_bytes.decode('utf-8')}

    def export(self):
        return {'key':self.key,'source_path':self.source_path,'revision':self.revision,
                'source_yaml':self.source_bytes.decode('utf-8'),'definition':self.definition}


def _independent_tree(value, active=None, depth=0, budget=None):
    # YAML aliases may share mutable containers. A targeted path edit must not
    # mutate another alias occurrence. Recursive YAML is not a finite document.
    if depth > 48: raise ValueError("material document nesting exceeds 48")
    budget=[100000] if budget is None else budget
    budget[0]-=1
    if budget[0]<0: raise ValueError("material document expansion exceeds limit")
    if isinstance(value,float) and not math.isfinite(value): raise ValueError("material values must be finite")
    active=set() if active is None else active
    if not isinstance(value,(dict,list)):
        if value is not None and type(value) not in (str,int,float,bool):
            raise ValueError("material definition requires JSON-compatible scalar values")
        return copy.deepcopy(value)
    if id(value) in active:raise ValueError('recursive YAML material is unsupported')
    active.add(id(value))
    try:
        if isinstance(value,dict):
            if any(not isinstance(k,str) for k in value):
                raise ValueError("material definition mapping keys must be strings")
            return {k:_independent_tree(v,active,depth+1,budget) for k,v in value.items()}
        return [_independent_tree(v,active,depth+1,budget) for v in value]
    finally:active.remove(id(value))


def _child(node,key):
    if isinstance(node,dict):
        if key not in node:raise KeyError(key)
        return node[key]
    if isinstance(node,list) and type(key) is int and 0<=key<len(node):
        return node[key]
    raise KeyError(key)


def read_material_document(key, directory=MATERIAL_DIRECTORY):
    key=_key(key);path=Path(directory).resolve()/(key+'.yaml')
    return MaterialDocument(key,str(path),path.read_bytes())


def read_material_documents(directory=MATERIAL_DIRECTORY):
    return tuple(read_material_document(p.stem,directory)
                 for p in sorted(Path(directory).glob('*.yaml')))


@dataclass(frozen=True)
class NativeMaterialPayload:
    key: str
    revision: str
    route: str
    payload: dict[str,Any]
    uncompiled_fields: tuple[str,...]
    warnings: tuple[str,...]
    source_snapshot: dict[str,Any] | None = None


def restore_material_document(snapshot, overrides=()):
    if not isinstance(snapshot,dict) or set(snapshot)!={'key','revision','source_yaml'}:
        raise ValueError('invalid canonical material snapshot')
    if not isinstance(snapshot['source_yaml'],str):
        raise ValueError('material source_yaml must be text')
    doc=MaterialDocument(snapshot['key'],'material-library:'+_key(snapshot['key']),
                         snapshot['source_yaml'].encode('utf-8'))
    if doc.revision!=snapshot['revision']:
        raise ValueError('canonical material source revision mismatch')
    if not isinstance(overrides,(tuple,list)):
        raise ValueError('material overrides must be a sequence')
    for override in overrides:
        if not isinstance(override,dict) or set(override)!={'path','value'}:
            raise ValueError('invalid targeted material override')
        doc=doc.with_field(override['path'],override['value'])
    return doc


def _nested_uncompiled(source):
    """Report fields outside the actual native record readers, with full paths."""
    schemas = {
        'pbr': set('albedo albedo_rgb roughness metallic transmission ior opacity emission emission_rgb'.split()),
        'enamel': set('thickness_nm thickness_m ior_real ior_imag roughness color_rgb'.split()),
        'texture_stack': set('emit_uv_layer color_uv_layer depth_uv_layer depth_layer remit_uv_layer remit_mask_layer depth_scale_mm thickness_scale_mm depth_bias_mm thickness_bias_mm emit_gain color_blend direct_lobe_power model_flags texture_flags flags remit_gain bulb_radius_mm remit_decay translucence_gain'.split()),
    }
    band_keys=set('center_hz center bandwidth_hz bandwidth reflectance reflectivity transmittance diffuse_frac diffusion emission reemission ior_real ior ior_imag'.split())
    result=[]
    def inspect(value, keys, path):
        if not isinstance(value,dict):
            result.append(path+' (expected mapping)'); return
        for key, child in value.items():
            field=path+'.'+str(key)
            if key not in keys: result.append(field)
            elif isinstance(child,dict): result.append(field+' (unsupported nested mapping)')
            elif isinstance(child,list) and any(isinstance(v,(dict,list)) for v in child):
                result.append(field+' (unsupported nested sequence)')
    for key,keys in schemas.items():
        if key in source: inspect(source[key],keys,key)
    for key in ('bands','spectral_bands'):
        if key in source and isinstance(source[key],list):
            for i,band in enumerate(source[key]): inspect(band,band_keys,f'{key}[{i}]')
    return result


def _validate_native_projection(payload):
    """Validate real float32 records before touching the shared registry."""
    import numpy as np
    from material_db import MaterialDatabase
    isolated=MaterialDatabase()
    isolated.register('validate',payload)
    try:
        with np.errstate(over='raise',invalid='raise'):
            tensors=isolated.build_tensors()
            for name,value in tensors.items():
                if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.number):
                    if not np.isfinite(value).all():
                        raise ValueError(f'nonfinite native material tensor: {name}')
    except (OverflowError,FloatingPointError) as exc:
        raise ValueError('material values overflow native float32 projection') from exc


def native_payload(document, *, profile_database=None):
    """Normalize source spellings into the existing native database adapters.

    Explicit legacy ray coefficients and authored PBR fields are both retained.
    MaterialDatabase derives only missing fields. Profile names must already be
    loaded with spectral_library.load_library. No file or source is mutated.
    """
    from material_db import EmissionProfileDatabase, MAX_SPECTRAL_BANDS
    from abstract_ui_material_catalog import _material_payload
    source=document.definition;warnings=[];uncompiled=[]
    # This cold spelling normalizer does not apply the matte catalogue policy.
    out=_material_payload(copy.deepcopy(source))
    nested=source.get('pbr',{}) if isinstance(source.get('pbr'),dict) else {}
    if 'albedo' not in nested and 'albedo_rgb' not in nested and 'albedo' in source:
        out['albedo']=copy.deepcopy(source['albedo'])
    elif 'albedo' in nested or 'albedo_rgb' in nested:
        out['albedo']=copy.deepcopy(out['albedo_rgb'])
    if 'emission' in nested and 'emission_rgb' not in nested:
        out['emission_rgb']=copy.deepcopy(nested['emission'])
    elif not any(k in nested for k in ('emission','emission_rgb')) and 'emission' in source and 'emission_rgb' not in source:
        out['emission_rgb']=copy.deepcopy(source['emission'])
    required=('reflectivity','diffusion','absorption')
    legacy=any(k in source for k in required)
    if legacy:
        if not all(k in source for k in required):
            raise ValueError('partial legacy ray coefficients need explicit completion')
        out['_source']='mat11'
        out['refl_in']=[float(source[k]) for k in required]
        out['refl_out']=out['refl_in'].copy()
    route='legacy+authored-pbr' if legacy else 'pbr'
    if 'emission_profile_name' in source:
        alias=source['emission_profile_name']
        if 'emit_profile_name' in source and source['emit_profile_name']!=alias:
            raise ValueError('conflicting emission profile names')
        source['emit_profile_name']=alias
    for name_key,idx_key in (('color_profile_name','color_profile_idx'),('emit_profile_name','emit_profile_idx'),('remit_profile_name','remit_profile_idx')):
        if idx_key in source and float(source[idx_key]) >= 0 and name_key not in source:
            raise ValueError(f'{idx_key} requires stable named dependency {name_key}; process-local indices are not portable authoring')
    ep=profile_database if profile_database is not None else EmissionProfileDatabase.instance()
    for name_key,idx_key in (('color_profile_name','color_profile_idx'),('emit_profile_name','emit_profile_idx'),('remit_profile_name','remit_profile_idx')):
        if name_key in source:
            name=source[name_key]
            if not isinstance(name,str) or name not in ep:
                raise ValueError(f'unresolved {name_key}: {name!r}')
            out[idx_key]=ep.index_of(name)
    flags=struct.unpack('I',struct.pack('f',float(out.get('mat_flags',0.))))[0]
    from mat_flags import MAT_FLAG_EMISSIVE, MAT_FLAG_REACTIVE
    if out.get('emit_profile_idx',-1)>=0:flags |= MAT_FLAG_EMISSIVE
    if out.get('reactive_shift_hz',0)!=0:flags |= MAT_FLAG_REACTIVE
    out['mat_flags']=struct.unpack('f',struct.pack('I',flags))[0]
    bands=source.get('bands',source.get('spectral_bands',[]))
    if not isinstance(bands,list) or len(bands)>MAX_SPECTRAL_BANDS:
        raise ValueError('spectral bands exceed native capacity or are not a list')
    known={'name','description','notes','gameplay','domain','albedo','albedo_rgb','pbr','roughness','smoothness','metallic','transmission','ior','opacity','emission','emission_rgb','ambient','spec_strength','shininess','grain','inner_color','bands','spectral_bands','enamel','texture_stack','reflectivity','diffusion','absorption','color_profile_name','emit_profile_name','emission_profile_name','remit_profile_name','color_profile_idx','emit_profile_idx','remit_profile_idx','reactive_shift_hz','mat_flags'}
    known.add('medium')
    uncompiled += [str(k) for k in source if k not in known]
    uncompiled += _nested_uncompiled(source)
    if 'medium' in source:
        # The authoritative document owns this cold definition. Surface PBR
        # records cannot represent volume density or molecular cross sections.
        uncompiled.append('medium (cold definition; explicit volume-field route)')
        warnings.append('Participating-medium optics are separate from the editor PBR placeholder; no extinction or scattering is inferred from opacity.')
    if not legacy:
        for field in ("reactive_shift_hz", "mat_flags"):
            if source.get(field, 0) != 0:
                uncompiled.append(field + " (not propagated by native PBR route)")
    _validate_native_projection(out)
    if not bands and 'medium' not in source:warnings.append('No authored spectral bands; native transport uses its existing fallback, not a measured spectrum.')
    return NativeMaterialPayload(document.key,document.revision,route,out,
                                 tuple(sorted(set(uncompiled))),tuple(warnings),
                                 document.snapshot())

def native_volume_payload(document, *, profile_database=None):
    """Explicit n/k-volume adapter; preserve but do not resolve surface-only data."""
    from dataclasses import replace
    source=document.definition
    if 'medium' in source:
        raise ValueError('participating-medium profiles require the explicit extinction-field route, not a surface n/k-volume fallback')
    ignored=[]
    for key in ('color_profile_name','color_profile_idx','enamel','texture_stack'):
        if key in source:
            ignored.append(key)
            source.pop(key)
    # Validate remaining active fields using the strict native adapter. In
    # particular named emit/remit dependencies are not dismissed as surface tint.
    projected=MaterialDocument(document.key,document.source_path,
        yaml.safe_dump(source,sort_keys=False).encode('utf-8'))
    compiled=native_payload(projected,profile_database=profile_database)
    payload=dict(compiled.payload)
    payload['_canonical_source_snapshot']=document.snapshot()
    payload['_volume_ignored_fields']=ignored
    warnings=compiled.warnings+tuple(f'{key}: preserved in source; ignored by native n/k volume transport' for key in ignored)
    return replace(compiled,revision=document.revision,payload=payload,
                   warnings=warnings,source_snapshot=document.snapshot())
