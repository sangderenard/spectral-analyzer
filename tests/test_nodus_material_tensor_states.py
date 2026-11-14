"""The Nodus state handoff uses the renderer's real material database rows."""
import hashlib

import numpy as np
import pytest

from canonical_material_documents import read_material_document, native_payload
from material_db import MaterialDatabase
from spectral_library import load_default_library


def test_canonical_and_studio_materials_share_exact_nodus_tensor_indices():
    profiles = load_default_library()
    document = read_material_document('borosilicate_glass')
    compiled = native_payload(document, profile_database=profiles)
    db = MaterialDatabase()
    db.register('saved_studio_canonical:borosilicate_glass:' + document.revision,
                compiled.payload, source_snapshot=compiled.source_snapshot)
    db.register('scene_local', {
        'albedo': [0.2, 0.3, 0.4], 'ior': 1.2,
    })
    grid = np.asarray([299792458.0 / 850e-9, 299792458.0 / 550e-9])
    state = db.build_nodus_tensor_states(grid)
    manifest, tensors = state['manifest'], state['tensors']

    assert manifest['material_names'][0].startswith('saved_studio_canonical:borosilicate_glass:')
    assert manifest['material_sources'] == [document.snapshot(), None]
    assert manifest['transport_frequency_hz'] == grid.tolist()
    np.testing.assert_array_equal(tensors['mat_buf'], db.build_mat_buf(grid))
    np.testing.assert_array_equal(tensors['pbr'], db.build_tensors()['pbr'])
    assert manifest['profile_index_names'] == profiles._order
    profile_index = profiles.index_of('basic_led_display')
    assert manifest['profile_sources'][profile_index] == profiles.source_revision('basic_led_display')
    assert manifest['profile_sources'][profile_index]['section'] == 'emission'
    assert 'emission_profiles' not in tensors
    assert tensors['mat_buf'].shape == (2 * 32, 12)
    assert tensors['mat_buf'][0, 0] == np.float32(grid[0])
    assert tensors['mat_buf'][32, 0] == np.float32(grid[0])
    for name, value in tensors.items():
        assert not value.flags.writeable
        assert manifest['tensors'][name]['sha256'] == hashlib.sha256(value.tobytes()).hexdigest()


def test_reregistering_as_programmatic_drops_stale_canonical_identity():
    profiles = load_default_library()
    document = read_material_document('borosilicate_glass')
    db = MaterialDatabase()
    db.register('m', native_payload(document, profile_database=profiles).payload,
                source_snapshot=document.snapshot())
    assert db.material_source_snapshot('m') == document.snapshot()
    db.register('m', {'albedo': [0.1, 0.1, 0.1]})
    assert db.material_source_snapshot('m') is None


def test_nodus_handoff_rejects_profile_index_drift():
    profiles = load_default_library()
    document = read_material_document('basic_led_display')
    compiled = native_payload(document, profile_database=profiles)
    altered = dict(compiled.payload, emit_profile_idx=-1)
    db = MaterialDatabase()
    db.register('display', altered, source_snapshot=document.snapshot())
    with pytest.raises(ValueError, match='disagrees with named source profile'):
        db.build_nodus_tensor_states(np.asarray([299792458.0 / 550e-9]))
