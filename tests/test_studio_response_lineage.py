"""Identity-only CPU checks; no render, GL context, or geometric lookup."""
import numpy as np
import pytest

from camera_designer.studio_response_lineage import NativeResponseLineage, copied_recipient, recipient_id
from camera_designer.studio_sphere_regions import pack_sphere_regions


def test_retained_native_rows_keep_distinct_subject_and_fixture_ordinals():
    lineage = NativeResponseLineage(2)
    lineage.append(1, 'subject', 2, np.asarray([0, 2, 4]))
    lineage.append(2, 'fixtures', 5, np.asarray([0, 3]))
    value = lineage.snapshot(geometry_sha256='a' * 64, revision=7)
    assert copied_recipient(value, 0) == (0, -1)
    assert copied_recipient(value, 2) == (1, 0)
    assert copied_recipient(value, 5) == (2, 0)
    assert copied_recipient(value, 6) == (2, 3)
    assert copied_recipient(value, 2, 2, 2) == (0, -1)
    np.testing.assert_array_equal(value['rows'][2:5, 1], [0, 2, 4])
    value['rows'][:] = 99
    assert lineage.rows[2, 1] == 0


def test_exact_sphere_compaction_copies_actual_selection_without_facet_guess():
    geom = np.zeros((7, 16), np.float32)
    geom[:, 4] = 1
    geom[:, 9] = 1
    shade = np.zeros_like(geom)
    tris = np.zeros((7, 3, 3), np.float32)
    scene = (geom, shade, np.zeros((32, 12), np.float32), tris,
             np.zeros((0, 8), np.float32), {}, (np.full(3, -1), np.ones(3)),
             np.zeros((0, 12), np.float32))
    original = np.asarray([0, 2, 3, 4])
    lineage = NativeResponseLineage(2)
    lineage.append(1, 'subject', 2, original)
    lineage.append(2, 'fixtures', 6, np.asarray([8]))
    regions = dict(version=1, spheres=[dict(start=2, count=3, center=[0, 0, 0], radius=.1)])
    plain, spheres_plain = pack_sphere_regions(scene, original, 2, regions, scale_m=1, center_m=[0, 0, 0])
    copied, spheres = pack_sphere_regions(scene, original, 2, regions, scale_m=1, center_m=[0, 0, 0], response_lineage_out=lineage)
    for a, b in zip(plain[:5], copied[:5]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(spheres, spheres_plain)
    value = lineage.snapshot()
    np.testing.assert_array_equal(value['spheres'], [[0, 3, 2, 3]])
    np.testing.assert_array_equal(value['rows'], [[0, -1, 0, -1], [0, -1, 0, -1],
                                               [1, 0, 0, -1], [1, 2, 1, 0], [2, 8, 0, -1]])
    assert copied_recipient(value, 3, 1, 0) == (3, 0)
    with pytest.raises(ValueError, match='kind'):
        copied_recipient(value, 3)
    with pytest.raises(ValueError, match='sphere event'):
        copied_recipient(value, 3, 1, 1)


def test_actual_packer_copies_existing_degenerate_filter_only_when_requested():
    from camera_designer.scene_builder import append_saved_studio_mesh_to_gpu_scene
    empty = (np.zeros((0, 16), np.float32), np.zeros((0, 16), np.float32),
             np.zeros((0, 12), np.float32), np.zeros((0, 3, 3), np.float32),
             np.zeros((0, 8), np.float32), {},
             (np.full(3, -1, np.float32), np.ones(3, np.float32)), np.zeros((0, 12), np.float32))
    faces = np.asarray([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
                        [[0, 0, 0]] * 3,
                        [[-1, -1, 1], [1, -1, 1], [0, 1, 1]]], np.float32)
    pbr = np.zeros((1, 16), np.float32)
    pbr[0, 7] = 1
    plain, retained = append_saved_studio_mesh_to_gpu_scene(empty, faces, [0, 0, 0], pbr,
        scale_m=1, center_m=[0, 0, 0], return_index_map=True)
    lineage = NativeResponseLineage(0)
    copied = append_saved_studio_mesh_to_gpu_scene(empty, faces, [0, 0, 0], pbr,
        scale_m=1, center_m=[0, 0, 0], response_lineage_out=lineage, response_source=(1, 'subject'))
    np.testing.assert_array_equal(retained, [0, 2])
    np.testing.assert_array_equal(lineage.rows[:, 1], retained)
    for a, b in zip(plain[:5], copied[:5]):
        np.testing.assert_array_equal(a, b)


def test_identity_copy_refuses_wrong_native_offsets_or_reordered_selection():
    lineage = NativeResponseLineage(2)
    with pytest.raises(ValueError):
        lineage.append(1, 'subject', 0, [0])
    with pytest.raises(ValueError):
        lineage.append(1, 'subject', 2, [2, 0])


def test_exact_wire_identity_and_required_opt_in():
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    assert recipient_id(0, -1) == 0
    assert recipient_id(1, 0) == 0x10000001
    assert recipient_id(2, 0) == 0x20000001
    assert recipient_id(3, 7) == 0x30000008
    for namespace, index in [(4, 0), (1, -1), (1, .5), (1, 0x0fffffff)]:
        with pytest.raises(ValueError):
            recipient_id(namespace, index)
    for config in ({}, {'light_field_information': 'spectral-arrivals-v1'}):
        with pytest.raises(ValueError, match='opt-in'):
            prepare_photo_geometry(config, None, None, None, response_lineage_out={})
    with pytest.raises(ValueError, match='unknown copied'):
        copied_recipient(NativeResponseLineage(1).snapshot(), 0, 3)


def test_photo_prepare_opt_in_copies_without_changing_scene_buffers():
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    vertices = np.asarray([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], np.float32)
    pbr = np.zeros((1, 16), np.float32)
    pbr[0, 7] = 1
    parts = {'version': 'studio-response-lineage-v1',
             'parts': [{'source_id': 'table/pane', 'start': 0, 'count': 1}]}
    fixtures = {'vertices': (vertices * .025 + [0, 0, .25]).reshape(-1).tolist(),
                'material_ids': [0], 'pbr': pbr.reshape(-1).tolist(),
                'response_lineage': parts}
    config = {'transport': 'fixed:3', 'fixtures': fixtures, 'geometry_sha256': 'a' * 64,
              'light_field_information': 'evaluated-responses-v1'}
    plain = prepare_photo_geometry(config, vertices, [0], pbr)
    lineage = {}
    copied = prepare_photo_geometry(config, vertices, [0], pbr, response_lineage_out=lineage)
    for index in (0, 1, 2, 3, 4, 7):
        np.testing.assert_array_equal(plain[0][index], copied[0][index])
    np.testing.assert_array_equal(plain[1], copied[1])
    assert lineage['rows'].shape == (len(plain[0][0]), 4)
    assert lineage['fixture_parts'] == parts
    assert lineage['fixture_parts'] is not parts
    assert lineage['geometry_sha256'] == 'a' * 64
    assert copied_recipient(lineage, len(lineage['rows']) - 1) == (2, 0)
