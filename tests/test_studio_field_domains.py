"""CPU-only native field contract tests: no renderer, context, or exposure."""
import hashlib
import copy

import numpy as np
import pytest

from camera_designer.studio_field_domains import (
    canonical_domain_key, capture_scene_contract, domain_sections, validate_scene_section,
)
from camera_designer.studio_rod_bundle_gpu import pack_bundle_records


def scene():
    # Host, dye, two rods, bubble, and a standalone transparent scene object.
    # Distinct IDs deliberately do not resemble authored material positions.
    geom = np.zeros((6, 16), np.float32)
    geom[:, 3] = [-1, -7, -10, -14, -20, 1]
    geom[:, 7] = [5, 9, 6, 7, 1, 8]
    geom[0, 11] = 4097
    geom[:, 0] = [0, 0, 0, 0, 0, 100]
    geom[:, 4] = 1
    geom[:, 9] = 1
    geom[:, 14] = 1
    shade = np.zeros((6, 16), np.float32)
    shade[:, 12:14] = -1
    shade[:, 14] = np.asarray([3, 7, 4, 5, 3, 6], np.uint32).view(np.float32)
    mats = np.zeros((8 * 32, 12), np.float32)
    mats[::32, 0] = 299792458. / 550e-9
    mats[::32, 7] = 1.5
    mats[7 * 32, 8] = .00001
    pbr = np.zeros((8, 16), np.float32)
    pbr[:, 5] = 1
    pbr[:, 6] = 1.5
    nodes = np.asarray([[-1, -1, -1, -1, 102, 2, 2, -1, 0, 6, 0, 0]], np.float32)
    ids = np.asarray([5, 1, 3, 0, 2, 4], np.int32)
    regions = np.asarray([[0, 3, 0, 0], [2, 7, 0, 0], [1, 4, 0, 0],
                          [1, 5, 0, 0], [3, -1, 0, 0], [4, 6, 0, 0]], np.float32)
    return dict(packed_geom=geom, packed_shade=shade, mat_buf=mats,
                bvh_nodes=nodes, bvh_ids=ids, context_buf=np.zeros((0, 8), np.float32),
                native_pbr=pbr, region_data=regions,
                bounds=([-1, -1, -1], [102, 2, 2]))


def section_array(payload, meta, name, origin=0):
    section = meta['buffers'][name]
    dtype = {'f32le': '<f4', 'i32le': '<i4'}[section['format']]
    start = section['offset_bytes'] - origin
    return np.frombuffer(payload[start:start + section['byte_length']], dtype=dtype).reshape(section['shape'])


def test_whole_native_scene_roundtrips_without_roi_or_material_remapping():
    data = scene()
    original = {k: v.copy() for k, v in data.items() if isinstance(v, np.ndarray)}
    contract = capture_scene_contract(**data)
    payload, meta = domain_sections(contract, 256)
    for section, key in [('geometry', 'packed_geom'), ('shade', 'packed_shade'),
                         ('bvh_nodes', 'bvh_nodes'), ('bvh_ids', 'bvh_ids'),
                         ('spectral_materials', 'mat_buf'), ('native_pbr', 'native_pbr'),
                         ('regions', 'region_data'), ('scale_contexts', 'context_buf')]:
        actual = section_array(payload, meta, section, 256)
        assert actual.tobytes() == original[key].tobytes()
        assert meta['buffers'][section]['sha256'] == hashlib.sha256(actual.tobytes()).hexdigest()
    assert section_array(payload, meta, 'geometry', 256)[5, 0] == 100
    domains = section_array(payload, meta, 'triangle_domains', 256)
    np.testing.assert_array_equal(domains[:, 1], [3, 7, 4, 5, 3, 6])
    np.testing.assert_array_equal(domains[:, 5:7], [[0, 0], [1, 2], [2, 1], [3, 1], [4, 3], [5, 4]])
    np.testing.assert_array_equal(domains[5, 3:5], [-1, 6])
    np.testing.assert_array_equal(domains[:5, 3:5], -2)
    lookup = section_array(payload, meta, 'medium_materials', 256)
    np.testing.assert_array_equal(lookup[9], [9, 7, 7, 7])
    assert meta['audit']['optical_buffers_complete']
    assert meta['audit']['convergence_certification'] == 'none'


def test_scene_capture_is_detached_and_serialization_is_relocatable():
    data = scene()
    contract = capture_scene_contract(**data)
    first, one = domain_sections(contract, 0)
    data['mat_buf'][:] = 99
    data['packed_geom'][:] = 99
    second, two = domain_sections(contract, 120)
    assert first == second
    assert one['scene_buffers_sha256'] == two['scene_buffers_sha256']
    for key in one['buffers']:
        assert two['buffers'][key]['offset_bytes'] == one['buffers'][key]['offset_bytes'] + 120
    one['audit']['volume_domains_complete'] = False
    assert domain_sections(contract, 0)[1]['audit']['volume_domains_complete']


def test_exact_domain_identity_distinguishes_same_medium_dyes_and_regions():
    plain = canonical_domain_key(3, [1], 6)
    dye = canonical_domain_key(3, [3], 6)
    other = canonical_domain_key(3, [5], 6)
    assert len({plain, dye, other}) == 3
    np.testing.assert_array_equal(np.frombuffer(dye, dtype='<u4'), [3, 3])
    assert canonical_domain_key(3, [3] + [0] * 31, 6) == dye
    assert canonical_domain_key(-1, [], 0) != canonical_domain_key(-2, [], 0)
    assert len(canonical_domain_key(3, [0] * 32, 1024)) == 132
    for bits in ([65], [], [3, 0]):
        with pytest.raises(ValueError):
            canonical_domain_key(3, bits, 6)
    with pytest.raises(ValueError, match='absent'):
        canonical_domain_key(3, [3] + [1] + [0] * 30, 6)


def test_exact_sphere_and_axial_bundle_records_remain_canonical():
    data = scene()
    bundle = dict(regions=[2, 3], eye_length=.75, eye_curve=.18, twist=.9, phase=.2,
                  frame=dict(origin=[0, 0, 0], axes=np.eye(3).tolist(), scale=1))
    data['region_data'] = pack_bundle_records(data['region_data'], bundle,
                                             scale_m=.025, center_m=[0, 0, .18])
    data['sphere_data'] = np.asarray([[0, 0, .18, .025, 0, 0, 0, 0]], np.float32)
    data['packed_geom'][0, 4:7] = 0
    data['packed_geom'][0, 8:11] = 0
    payload, meta = domain_sections(capture_scene_contract(**data), 0)
    np.testing.assert_array_equal(section_array(payload, meta, 'regions'), data['region_data'])
    np.testing.assert_array_equal(section_array(payload, meta, 'spheres'), data['sphere_data'])
    assert meta['region_count'] == 6
    assert section_array(payload, meta, 'triangle_domains')[0, 7] == 0
    assert meta['buffers']['regions']['count'] == 11


def test_unsupported_transparency_and_parametric_flags_are_not_complete():
    data = scene()
    data['region_data'] = None
    data['packed_geom'][:, 3] = 0
    data['packed_geom'][1, 15] = np.asarray([32], np.uint32).view(np.float32)[0]
    payload, meta = domain_sections(capture_scene_contract(**data), 0)
    assert meta['audit']['untagged_transparent_triangle_ids'] == list(range(6))
    assert meta['audit']['unsupported_parametric_triangle_ids'] == [1]
    assert not meta['audit']['volume_domains_complete']
    assert np.all(section_array(payload, meta, 'triangle_domains')[:, 2] == 3)


def test_missing_spectra_and_profile_data_remain_explicit():
    data = scene()
    data['mat_buf'][7 * 32:(7 + 1) * 32] = 0
    data['packed_shade'][0, 12] = 0
    meta = capture_scene_contract(**data)['metadata']
    assert meta['audit']['missing_spectral_material_ids'] == [7]
    assert meta['audit']['missing_profile_triangle_ids'] == [0]
    assert not meta['audit']['optical_buffers_complete']
    data['emission_profiles'] = np.zeros((1, 20), np.float32)
    assert not capture_scene_contract(**data)['metadata']['audit']['missing_profile_triangle_ids']


def test_missing_or_partial_native_ids_and_buffers_are_rejected():
    for change, match in [
        (lambda d: d.update(bvh_ids=d['bvh_ids'][:-1]), 'every packed'),
        (lambda d: d.update(native_pbr=d['native_pbr'][:-1]), 'identity tables'),
        (lambda d: d.update(region_data=None), 'missing its native'),
        (lambda d: d['bvh_nodes'].__setitem__((0, 9), 5), 'omits'),
        (lambda d: d['packed_shade'].__setitem__((0, 14), np.asarray([99], np.uint32).view(np.float32)[0]), 'outside'),
    ]:
        data = scene()
        change(data)
        with pytest.raises(ValueError, match=match):
            capture_scene_contract(**data)


def test_no_resource_overflow_can_silently_omit_domains():
    contract = capture_scene_contract(**scene())
    payload, _ = domain_sections(contract, 0)
    assert len(domain_sections(contract, 120, 120 + len(payload))[0]) == len(payload)
    with pytest.raises(ValueError, match='no geometry or materials omitted'):
        domain_sections(contract, 120, 120 + len(payload) - 1)
    with pytest.raises(ValueError, match='aligned'):
        domain_sections(contract, 3)


def test_bit_packed_native_flags_are_preserved_even_when_float_is_nan():
    data = scene()
    data['packed_geom'][0, 15] = np.asarray([0xffffffff], np.uint32).view(np.float32)[0]
    payload, meta = domain_sections(capture_scene_contract(**data), 0)
    assert section_array(payload, meta, 'geometry')[0, 15:].copy().view(np.uint32)[0] == 0xffffffff


def test_actual_scene_packer_optics_and_native_ids_export_without_reconstruction():
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    from camera_designer.studio_region_gpu import region_records_from_geometry
    from material_db import MaterialDatabase

    empty = (np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32),
             np.zeros((0, 12), np.float32), np.zeros((0, 3, 3), np.float32),
             np.zeros((0, 8), np.float32), {},
             (np.full(3, -1, np.float32), np.ones(3, np.float32)),
             np.zeros((0, 12), np.float32))
    vertices = np.asarray([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]] * 2, np.float32)
    pbr = np.zeros((2, 16), np.float32)
    pbr[:, 5] = 1
    pbr[:, 6] = 1.52
    spec = {'version': 1, 'regions': [
        dict(start=0, count=1, mode='host', material=0),
        dict(start=1, count=1, mode='injection_dye', material=1)]}
    optics = {'version': 1, 'materials': [
        dict(material=i, wavelengths_nm=[400, 700], transmittance=[.7, .7],
             reference_thickness_mm=1, concentration=1) for i in range(2)]}
    native = append_saved_studio_mesh_to_gpu_scene(
        empty, vertices, [0, 1], pbr, scale_m=.01, center_m=[0, 0, 0],
        wavelengths_um=[.65, .55, .45], region_media=spec, optical_media=optics)
    geom, shade, mats, tris, contexts, _, bounds, _ = native
    nodes = np.asarray([[*tris.min(axis=(0, 1)), -1, *tris.max(axis=(0, 1)), -1,
                         0, len(geom), 0, 0]], np.float32)
    contract = capture_scene_contract(
        geom, shade, mats, nodes, np.arange(len(geom)), contexts,
        MaterialDatabase.instance().build_tensors()['pbr'],
        region_data=region_records_from_geometry(geom), bounds=bounds,
        transport_source='actual-assembled-program-for-source-hash-test')
    payload, meta = domain_sections(contract, 0)
    assert meta['audit']['volume_domains_complete']
    # The isolated CPU packer fixture does not provide its profile upload.
    # Preserve that missing input rather than pretending a full program exists.
    assert not meta['audit']['optical_buffers_complete']
    assert meta['audit']['missing_profile_triangle_ids']
    assert section_array(payload, meta, 'spectral_materials').tobytes() == mats.tobytes()
    assert meta['transport_program_sha256'] == hashlib.sha256(
        b'actual-assembled-program-for-source-hash-test').hexdigest()


def test_delivered_scene_validator_uses_exact_native_bytes_and_absolute_ranges():
    origin = 4096
    payload, meta = domain_sections(capture_scene_contract(**scene()), origin)
    reads = []

    def read_range(offset, count):
        reads.append((offset, count))
        return payload[offset - origin:offset - origin + count]

    checked = validate_scene_section(meta, read_range)
    assert checked == meta and checked is not meta
    assert len(reads) == len(meta['buffers'])
    # Different receiving-source revisions must not rewrite producer identity.
    historic = copy.deepcopy(meta)
    historic['source_sha256'] = dict.fromkeys(meta['source_sha256'], 'a' * 64)
    historic['transport_program_sha256'] = 'b' * 64
    assert validate_scene_section(historic, read_range) == historic


@pytest.mark.parametrize('mutation', ['hash', 'audit', 'identity', 'layout', 'tail', 'source'])
def test_delivered_scene_validator_rejects_corruption_and_false_completeness(mutation):
    payload, meta = domain_sections(capture_scene_contract(**scene()), 0)
    if mutation == 'hash':
        payload = b'xxxx' + payload[4:]
    elif mutation == 'audit':
        meta['audit']['volume_domains_complete'] = False
    elif mutation == 'identity':
        s = meta['buffers']['triangle_domains']
        start = s['offset_bytes']
        payload = payload[:start] + np.asarray([99], '<i4').tobytes() + payload[start + 4:]
        s['sha256'] = hashlib.sha256(payload[start:start + s['byte_length']]).hexdigest()
    elif mutation == 'layout':
        meta['buffers']['spheres']['offset_bytes'] += 4
    elif mutation == 'tail':
        meta['byte_length'] += 4
        payload += b'xxxx'
    elif mutation == 'source':
        meta['transport_program_sha256'] = 'invented'
    with pytest.raises(ValueError):
        validate_scene_section(meta, lambda offset, count: payload[offset:offset + count])


def test_delivered_scene_validator_preserves_known_missing_domain_state():
    data = scene()
    data['packed_geom'][:, 3] = 0
    data['region_data'] = None
    payload, meta = domain_sections(capture_scene_contract(**data), 0)
    checked = validate_scene_section(meta, lambda offset, count: payload[offset:offset + count])
    assert not checked['audit']['volume_domains_complete']
    meta['audit']['volume_domains_complete'] = True
    with pytest.raises(ValueError, match='audit'):
        validate_scene_section(meta, lambda offset, count: payload[offset:offset + count])
