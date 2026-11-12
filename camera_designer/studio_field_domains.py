"""CPU-only export of the actual native scene used by opted-in field recording.

This is a transport contract, not an intersector or a material approximation.
The caller must opt in before capture; ordinary tracing never calls this module.
All packed rows retain their native IDs and bit patterns. No recording-ROI cut
is applied to scene geometry, optical materials, or region ownership.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

import numpy as np

from mat_flags import (MAX_SPECTRAL_BANDS, MAT_FLAG_PARAMETRIC,
                       MAT_FLAG_TRANSMISSIVE)
from .studio_region_gpu import validate_region_records
from .studio_rod_bundle_gpu import region_record_count

VERSION = 'native-scene-domains-v1'
PAYLOAD_LIMIT = 256 * 1024 * 1024
TRIANGLE_DOMAIN_CHANNELS = [
    'triangle_id', 'native_material_id', 'ownership_kind',
    'positive_medium', 'negative_medium', 'region_id', 'region_mode',
    'sphere_index',
]
_BUFFER_SPECS = {
    'geometry': ('f32le', 16, 0), 'shade': ('f32le', 16, 9),
    'bvh_nodes': ('f32le', 12, 1), 'bvh_ids': ('i32le', 1, 2),
    'spectral_materials': ('f32le', 12, 10), 'scale_contexts': ('f32le', 8, 8),
    'native_pbr': ('f32le', 16, 18), 'spheres': ('f32le', 8, 12),
    'regions': ('f32le', 4, 15), 'emission_profiles': ('f32le', 1, 7),
    'triangle_domains': ('i32le', 8, None), 'medium_materials': ('i32le', 4, None),
}


def _rows(value, width, name, *, bit_columns=()):
    a = np.array(value, dtype='<f4', copy=True, order='C')
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f'{name} must have {width} native float32 columns')
    columns = [i for i in range(width) if i not in bit_columns]
    if not np.isfinite(a[:, columns]).all():
        raise ValueError(f'{name} contains nonfinite numeric values')
    return a


def _integer_values(values, name, low, high):
    a = np.asarray(values)
    if np.any(a != np.floor(a)) or np.any(a < low) or np.any(a > high):
        raise ValueError(f'{name} contains an invalid native index')
    return a.astype(np.int64)


def _validate_bvh(nodes, ids, triangle_count):
    """Validate the supplied tree, never rebuild/reorder it during export."""
    if len(ids) != triangle_count or not np.array_equal(np.sort(ids), np.arange(triangle_count)):
        raise ValueError('BVH IDs must cover every packed scene triangle exactly once')
    if not triangle_count:
        if len(nodes):
            raise ValueError('empty geometry has a nonempty BVH')
        return
    if not len(nodes):
        raise ValueError('whole-scene BVH is missing')
    links = _integer_values(nodes[:, [3, 7, 8, 9]], 'BVH links', -1, max(len(nodes), len(ids)))
    pending = [0]
    seen = set()
    covered = np.zeros(len(ids), dtype=bool)
    while pending:
        index = pending.pop()
        if index in seen or not 0 <= index < len(nodes):
            raise ValueError('BVH has repeated, cyclic, or out-of-range nodes')
        seen.add(index)
        left, right, start, count = links[index]
        if np.any(nodes[index, :3] > nodes[index, 4:7]):
            raise ValueError('BVH bounds are reversed')
        if count > 0:
            if left != -1 or right != -1 or start < 0 or start + count > len(ids):
                raise ValueError('invalid BVH leaf')
            if np.any(covered[start:start + count]):
                raise ValueError('overlapping BVH leaf coverage')
            covered[start:start + count] = True
        else:
            if count != 0 or left < 0 or right < 0:
                raise ValueError('invalid BVH branch')
            pending.extend([int(right), int(left)])
    if len(seen) != len(nodes) or not np.all(covered):
        raise ValueError('BVH omits scene nodes or triangle IDs')


def canonical_domain_key(medium, active_words, region_count):
    """Collision-free domain identity: signed medium plus canonical active bits.

    This is not a hash. In a region scene, equal medium IDs alone are NEVER
    equal domains. Active host/dye/rod/air/standalone bits remain distinguishable.
    The producer must accumulate different keys separately or report unresolved.
    """
    if type(medium) is not int or medium < -2 or medium > 2**31 - 1:
        raise ValueError('invalid canonical medium ID')
    if type(region_count) is not int or not 0 <= region_count <= 1024:
        raise ValueError('invalid canonical region count')
    a = np.asarray(active_words)
    count = (region_count + 31) // 32
    if a.ndim != 1 or len(a) not in (count, 32):
        raise ValueError('domain key requires the complete canonical active bitset')
    if (not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 2**32 - 1)
            or np.any(a != np.floor(a))):
        raise ValueError('invalid active region words')
    words = a.astype('<u4')
    if np.any(words[count:]) or (region_count % 32 and count and
            int(words[count - 1]) >> (region_count % 32)):
        raise ValueError('active bitset references absent regions')
    return np.asarray([medium], dtype='<i4').tobytes() + words[:count].tobytes()


def _source_contract():
    root = Path(__file__).resolve().parent.parent
    paths = ['demo_pluck_gl.py', 'mat_flags.py', 'material_db.py',
             'camera_designer/studio_boundary_media.py',
             'camera_designer/studio_region_gpu.py',
             'camera_designer/studio_sphere_regions.py',
             'camera_designer/parametric_surfaces.py',
             'camera_designer/studio_rod_bundle_gpu.py',
             'camera_designer/studio_conductor_surface.py',
             'camera_designer/studio_opaque_pbr.py']
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def capture_scene_contract(packed_geom, packed_shade, mat_buf, bvh_nodes,
                           bvh_ids, context_buf, native_pbr, sphere_data=None,
                           region_data=None, *, bounds, surface_transport=None,
                           emission_profiles=None, transport_source=None):
    """Detach and audit full native buffers, with no GL import or allocation.

    Called only for explicit spectral-arrivals-v1 information requests, after
    the canonical prebuilt scene is assembled, and again on geometry updates.
    Returns immutable byte parts plus detached metadata, ready for relocation
    into the same field artifact. Unsupported volume information is explicit;
    copying geometry is not certification that every consumer can evaluate it.
    """
    if surface_transport not in (None, 'studio-pbr-mixture-v1'):
        raise ValueError('unsupported native surface transport')
    geom = _rows(packed_geom, 16, 'TriGeom', bit_columns=(15,))
    shade = _rows(packed_shade, 16, 'TriShade', bit_columns=(14,))
    mats = _rows(mat_buf, 12, 'MatBuf')
    pbr = _rows(native_pbr, 16, 'NativePBR')
    nodes = _rows(bvh_nodes, 12, 'NodeBuf')
    contexts = _rows(context_buf, 8, 'ScaleContextBuf')
    spheres = _rows(np.empty((0, 8)) if sphere_data is None else sphere_data, 8, 'StudioSphereBuf')
    if len(shade) != len(geom) or len(mats) % MAX_SPECTRAL_BANDS:
        raise ValueError('native triangle/material row counts disagree')
    material_count = len(mats) // MAX_SPECTRAL_BANDS
    if len(pbr) != material_count:
        raise ValueError('native PBR and spectral material identity tables disagree')
    ids_raw = np.asarray(bvh_ids)
    if ids_raw.ndim != 1:
        raise ValueError('BVH IDs must be one-dimensional')
    ids = _integer_values(ids_raw, 'BVH triangle IDs', 0, max(len(geom) - 1, 0)).astype('<i4')
    _validate_bvh(nodes, ids, len(geom))
    limits = np.asarray(bounds, dtype='<f4')
    if limits.shape != (2, 3) or not np.isfinite(limits).all() or np.any(limits[1] <= limits[0]):
        raise ValueError('invalid whole-scene transport bounds')
    material_ids = shade[:, 14].copy().view('<u4')
    if np.any(material_ids >= material_count):
        raise ValueError('triangle material ID is outside the native material table')
    flags = geom[:, 15].copy().view('<u4')
    regions = (np.empty((0, 4), dtype='<f4') if region_data is None else
               np.array(validate_region_records(region_data, geom, shade, material_count), dtype='<f4', copy=True))
    count = region_record_count(regions)
    region_tagged = geom[:, 3] <= -1
    if np.any(region_tagged) and not count:
        raise ValueError('tagged region geometry is missing its native region buffer')
    pair_tagged = (geom[:, 3] >= 1) & (geom[:, 7] >= 1)
    if np.any(pair_tagged):
        _integer_values(geom[pair_tagged][:, [3, 7]] - 2, 'boundary media', -1, material_count - 1)
    sphere_ids = _integer_values(spheres[:, 4], 'sphere owner IDs', 0, max(len(geom) - 1, 0))
    if (len(spheres) > 65 or len(set(sphere_ids)) != len(sphere_ids)
            or np.any(spheres[:, 3] <= 0) or np.any(spheres[:, 5:] != 0)):
        raise ValueError('invalid native exact sphere records')
    if len(spheres) and (not len(geom) or np.any(geom[sphere_ids, 4:7] != 0)
                        or np.any(geom[sphere_ids, 8:11] != 0)):
        raise ValueError('exact sphere owner must retain canonical degenerate geometry')
    transparent = ((shade[:, 7] < .999) | ((flags & MAT_FLAG_TRANSMISSIVE) != 0)
                   | (pbr[material_ids, 5] > 0) | (pbr[material_ids, 7] < .999))
    enabled = bool(count or (len(geom) and geom[0, 11] == 4097))
    pair_active = pair_tagged & (shade[:, 7] < .999) & enabled
    region_active = region_tagged & (shade[:, 7] < .999) & bool(count)
    unresolved = transparent & ~(pair_active | region_active)
    domain = np.full((len(geom), 8), -2, dtype='<i4')
    domain[:, 0] = np.arange(len(geom))
    domain[:, 1] = material_ids
    domain[:, 2] = np.where(unresolved, 3, 0)
    domain[:, 5:] = -1
    domain[pair_active, 2] = 1
    domain[pair_active, 3:5] = geom[pair_active][:, [3, 7]] - 2
    domain[region_active, 2] = 2
    if count:
        codes = (-geom[region_active, 3] - 1).astype(np.int64)
        domain[region_active, 5] = codes // 4
        domain[region_active, 6] = codes % 4
        for ri in np.flatnonzero(regions[:count, 0] == 4):
            selected = pair_active & (geom[:, 7] - 2 == regions[ri, 1])
            domain[selected, 5:7] = [ri, 4]
    domain[sphere_ids, 7] = np.arange(len(spheres))
    # IDs are identities, not an authored-PBR reverse lookup or a color guess.
    medium = np.full((material_count + 2, 4), -1, dtype='<i4')
    medium[:, 0] = np.arange(material_count + 2)
    medium[:, 1] = np.arange(-2, material_count)
    medium[0, 2:] = -2
    medium[2:, 2:] = np.arange(material_count)[:, None]
    profiles = np.asarray([] if emission_profiles is None else emission_profiles, dtype='<f4')
    if not np.isfinite(profiles).all():
        raise ValueError('nonfinite native emission profile data')
    profiles = profiles.reshape(-1, 1).copy()
    profile_users = np.flatnonzero((shade[:, 11] > .5) | (shade[:, 12] >= 0) | (shade[:, 13] >= 0))
    missing_profiles = []
    for ti in profile_users:
        indices = shade[ti, 12:14]
        indices = indices[indices >= 0]
        if (not len(profiles) or np.any(indices != np.floor(indices))
                or np.any((indices + 1) * 20 > len(profiles))):
            missing_profiles.append(int(ti))
    referenced = set(int(i) for i in material_ids)
    for row in domain[pair_active, 3:5].ravel():
        if row >= 0:
            referenced.add(int(row))
    referenced.update(int(v) for v in regions[:count, 1] if v >= 0)
    inactive_spectra = [i for i in sorted(referenced) if not np.any(mats[i * MAX_SPECTRAL_BANDS:(i + 1) * MAX_SPECTRAL_BANDS, 0] > 0)]
    parametric = np.flatnonzero((flags & MAT_FLAG_PARAMETRIC) != 0)
    audit = dict(untagged_transparent_triangle_ids=np.flatnonzero(unresolved).tolist(),
                 unsupported_parametric_triangle_ids=parametric.tolist(),
                 missing_spectral_material_ids=inactive_spectra,
                 missing_profile_triangle_ids=missing_profiles,
                 geometry_topology_certification='native supplied topology; no new watertightness proof',
                 arrival_domain_certification='separate per-arrival canonical medium plus full active-region bitset required',
                 convergence_certification='none')
    audit['volume_domains_complete'] = not (np.any(unresolved) or len(parametric) or inactive_spectra)
    audit['optical_buffers_complete'] = audit['volume_domains_complete'] and not audit['missing_profile_triangle_ids']
    arrays = [
        ('geometry', geom, 'f32le', 0, 'TriGeom'),
        ('shade', shade, 'f32le', 9, 'TriShade'),
        ('bvh_nodes', nodes, 'f32le', 1, 'Node'),
        ('bvh_ids', ids.reshape(-1, 1), 'i32le', 2, 'native triangle ordering'),
        ('spectral_materials', mats, 'f32le', 10, 'SpectralBandRecord'),
        ('scale_contexts', contexts, 'f32le', 8, 'ScaleContext'),
        ('native_pbr', pbr, 'f32le', 18, 'native-pbr16'),
        ('spheres', spheres, 'f32le', 12, 'StudioSphereRegion'),
        ('regions', regions, 'f32le', 15, 'StudioRegionBuf including bundle records'),
        ('emission_profiles', profiles, 'f32le', 7, 'flat native EmissionProfileBuf'),
        ('triangle_domains', domain, 'i32le', None, 'explicit native ownership'),
        ('medium_materials', medium, 'i32le', None, 'encoded medium/native material identity'),
    ]
    parts = []
    digest = hashlib.sha256()
    for name, a, fmt, binding, layout in arrays:
        raw = a.tobytes(order='C')
        section = dict(count=len(a), stride_bytes=a.shape[1] * 4, format=fmt,
                       layout=layout, shape=list(a.shape), binding=binding,
                       byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        parts.append((name, raw, section))
        digest.update(name.encode('ascii') + b'\0')
        digest.update(np.asarray(a.shape, dtype='<u8').tobytes())
        digest.update(raw)
    digest.update(limits.tobytes())
    digest.update(str(surface_transport).encode('ascii'))
    metadata = dict(version=VERSION, scene_buffers_sha256=digest.hexdigest(),
        scene_scope='entire native transport scene; independent of recording ROI',
        bounds_min_m=limits[0].tolist(), bounds_max_m=limits[1].tolist(),
        material_count=material_count, max_spectral_bands=MAX_SPECTRAL_BANDS,
        region_count=count, active_region_word_count=(count + 31) // 32,
        region_modes={'host': 0, 'glass_rod': 1, 'injection_dye': 2, 'air_bubble': 3, 'standalone': 4},
        triangle_domain_channels=TRIANGLE_DOMAIN_CHANNELS,
        ownership_kinds={'surface_only': 0, 'explicit_pair': 1, 'dynamic_region': 2, 'unresolved_transparency': 3},
        medium_material_channels=['encoded_medium', 'medium_id', 'spectral_material_id', 'native_pbr_id'],
        medium_encoding='encoded = canonical medium + 2; 0 invalid, 1 air, >=2 exact native material identity',
        domain_key='i32le medium followed by ceil(region_count/32) u32le active-region words; no hash substitution',
        additive_dye='native region mode 2 contributes its MatBuf k only when both host region 0 and that dye region are active; base medium ID alone is insufficient',
        mixed_domain='different exact keys require separate accumulators; an unsplit mixed bin is unresolved, not black',
        spectral_channels=['center_hz', 'bandwidth_hz', 'reflectance_mag', 'transmittance', 'diffuse_frac', 'emission', 'reemission', 'ior_real', 'ior_imag', 'reactive_shift_hz', 'ggx_alpha', 'metallic'],
        spectral_lookup='canonical studio_medium_optics nearest positive-frequency row by absolute log-frequency ratio; n=max(row[7],1e-6), k=max(row[8],0)',
        optical_authority='studio_boundary_media + studio_region_gpu REGION_WRAPPERS; exact sphere and axial divider adapters',
        supported_volume_representations=['explicit native triangle medium pairs', 'tagged host/rod/dye/air region shells', 'native standalone regions', 'native exact spheres with owner triangles', 'native axial bundle affine/divider records'],
        unsupported_volume_representations=['untagged transparent solids', 'ParamSurf descriptor-only volumes', 'ExtinctionGrid combined with internal-media arrivals'],
        surface_transport=surface_transport, source_sha256=_source_contract(),
        source_hash_semantics='source file identities; actual assembled program is identified separately when supplied',
        transport_program_sha256=(hashlib.sha256(transport_source.encode('utf-8')).hexdigest()
                                  if transport_source is not None else None),
        audit=audit)
    return dict(parts=tuple(parts), metadata=metadata)


def domain_sections(contract, offset_bytes, payload_limit=PAYLOAD_LIMIT):
    """Place complete byte-exact scene sections; fail before any truncation."""
    if type(offset_bytes) is not int or offset_bytes < 0 or offset_bytes % 4:
        raise ValueError('scene section offset must be a nonnegative four-byte aligned integer')
    size = sum(len(raw) for _, raw, _ in contract['parts'])
    if size > PAYLOAD_LIMIT:
        raise ValueError('complete native scene exceeds 256 MiB section limit; no geometry or materials omitted')
    if offset_bytes + size > payload_limit:
        raise ValueError('complete native scene exceeds field payload limit; no geometry or materials omitted')
    metadata = copy.deepcopy(contract['metadata'])
    metadata.update(offset_bytes=offset_bytes, byte_length=size, buffers={})
    parts = []
    for name, raw, descriptor in contract['parts']:
        metadata['buffers'][name] = {**descriptor, 'offset_bytes': offset_bytes}
        offset_bytes += len(raw)
        parts.append(raw)
    return b''.join(parts), metadata


def validate_scene_section(metadata, read_range):
    """Validate delivered scene provenance and integrity, using native checks.

    ``read_range(absolute_offset, byte_length)`` must return exactly those bytes
    from the admitted logical package. This is server-side admission, not
    traversal, optical evaluation, or permission to reconstruct a second-pass
    response. No GL import, context, render, or external state is involved.
    Returns detached validated metadata. Unsupported domains remain explicit.
    """
    if not isinstance(metadata, dict) or metadata.get('version') != VERSION:
        raise ValueError('unsupported native scene contract version')
    offset, length = metadata.get('offset_bytes'), metadata.get('byte_length')
    if (type(offset) is not int or offset < 0 or offset % 4 or
            type(length) is not int or not 0 <= length <= PAYLOAD_LIMIT):
        raise ValueError('invalid native scene section bounds')
    buffers = metadata.get('buffers')
    if not isinstance(buffers, dict) or set(buffers) != set(_BUFFER_SPECS):
        raise ValueError('native scene buffer set is incomplete or unknown')
    cursor = offset
    arrays = {}
    raw_sections = {}
    for name, (fmt, width, binding) in _BUFFER_SPECS.items():
        descriptor = buffers[name]
        if not isinstance(descriptor, dict):
            raise ValueError('invalid native scene buffer descriptor')
        count = descriptor.get('count')
        if (type(count) is not int or count < 0 or
                descriptor.get('format') != fmt or descriptor.get('stride_bytes') != width * 4 or
                descriptor.get('shape') != [count, width] or descriptor.get('binding') != binding or
                descriptor.get('offset_bytes') != cursor or descriptor.get('byte_length') != count * width * 4):
            raise ValueError(f'invalid native scene buffer layout: {name}')
        size = count * width * 4
        if cursor + size > offset + length:
            raise ValueError('native scene buffer exceeds admitted section')
        raw = read_range(cursor, size)
        if not isinstance(raw, (bytes, bytearray, memoryview)) or len(raw) != size:
            raise ValueError(f'incomplete native scene buffer bytes: {name}')
        raw = bytes(raw)
        if hashlib.sha256(raw).hexdigest() != descriptor.get('sha256'):
            raise ValueError(f'native scene buffer digest mismatch: {name}')
        raw_sections[name] = raw
        arrays[name] = np.frombuffer(raw, dtype='<f4' if fmt == 'f32le' else '<i4').reshape(count, width)
        cursor += size
    if cursor != offset + length:
        raise ValueError('native scene section contains unaccounted bytes')
    captured = capture_scene_contract(
        arrays['geometry'], arrays['shade'], arrays['spectral_materials'],
        arrays['bvh_nodes'], arrays['bvh_ids'].reshape(-1), arrays['scale_contexts'],
        arrays['native_pbr'], arrays['spheres'],
        arrays['regions'] if len(arrays['regions']) else None,
        bounds=(metadata.get('bounds_min_m'), metadata.get('bounds_max_m')),
        surface_transport=metadata.get('surface_transport'), emission_profiles=arrays['emission_profiles'])
    checked = captured['metadata']
    for name, raw, _ in captured['parts']:
        if raw != raw_sections[name]:
            raise ValueError(f'native scene derived identity disagrees with buffers: {name}')
    # Source hashes describe the producer, not the receiving process's files.
    # Compare only structural/identity semantics to our native validator.
    for key, expected in checked.items():
        if key in ('source_sha256', 'transport_program_sha256'):
            continue
        if metadata.get(key) != expected:
            raise ValueError(f'native scene metadata disagrees with buffers: {key}')
    sources = metadata.get('source_sha256')
    if (not isinstance(sources, dict) or set(sources) != set(checked['source_sha256']) or
            any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                for value in sources.values())):
        raise ValueError('invalid native source provenance hashes')
    program = metadata.get('transport_program_sha256')
    if program is not None and (not isinstance(program, str) or not re.fullmatch('[0-9a-f]{64}', program)):
        raise ValueError('invalid assembled transport program hash')
    return copy.deepcopy(metadata)
