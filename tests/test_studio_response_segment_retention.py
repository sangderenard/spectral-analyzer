"""CPU-only retention of existing native records, never new ray transport."""
import numpy as np
import pytest

from camera_designer.studio_response_lineage import NativeResponseLineage
from camera_designer.studio_scene_responses import SceneResponseStore, validate_scene_response_snapshot


def store(*, bounds=((0, 0, 0), (1, 1, 1))):
    lineage = NativeResponseLineage(0)
    lineage.append(1, 'subject', 0, np.array([0]))
    return SceneResponseStore(scene_revision='physical', memory_budget_bytes=1024*1024,
        lineage_provider=lambda: lineage.snapshot(), bounds=bounds, dimensions=[32]*3)


def metadata(sequence=0):
    return dict(scene_revision='physical', stream_id='native', sequence=sequence,
        source_sha256='source', transport_scene_sha256='transport',
        primitive_identity_offsets=[112, 113])


def segment(start=(-1, .5, .5), end=(2, .5, .5), *, family=1, kind=5, primitive=0):
    row = np.zeros((1, 128), dtype='<u4')
    row[0, :2] = [kind, family]
    row[0, 5] = 0  # Actual native hit, not an inferred endpoint.
    row[0, 30] = 1
    row[0, 21:24] = np.asarray(end, np.float32).view(np.uint32)
    row[0, 27:30] = np.asarray([0, 1, 0], np.float32).view(np.uint32)
    row[0, 112:114] = [primitive, 0]
    delta = np.asarray(end, np.float64) - start
    direction = delta / np.linalg.norm(delta) if np.any(delta) else np.zeros(3)
    row[0, 114:117] = direction.astype(np.float32).view(np.uint32)
    row[0, 117:120] = np.asarray(start, np.float32).view(np.uint32)
    # Interior identity, zero response, unknown source bits and payload NaN
    # remain native evidence. None of these values decides spatial retention.
    row[0, 7:9] = [4, 6]
    row[0, 32] = 0x80000001
    row[0, 64] = 0x40000002
    row[0, 104] = 0x7fc12345
    row[0, 124] = 4
    return row


@pytest.mark.parametrize('family', [1, 2])
@pytest.mark.parametrize('kind', [5, 6])
@pytest.mark.parametrize('primitive', [0, 2])
def test_crossing_records_survive_exactly_without_endpoint_or_energy_rewrite(family, kind, primitive):
    s = store()
    rows = segment(family=family, kind=kind, primitive=primitive)
    s.consume_forward(metadata(), rows)
    manifest, payload = s.snapshot()
    assert s.retained_entries == 1 and s.omitted_out_of_scope == 0
    section = manifest['sections'][1]
    copied = np.frombuffer(payload, '<u4', count=128, offset=section['offset'])
    np.testing.assert_array_equal(copied, rows[0])
    index = np.frombuffer(payload, '<u4', count=20)
    assert index[0] == family and index[1] == kind
    assert index[11] == 1  # Actual endpoint exists, but is outside this ROI.
    assert np.all(index[16:19] == 0xffffffff)  # Never clamp it to a boundary cell.
    np.testing.assert_array_equal(index[8:11], rows[0, 21:24])
    assert not manifest['application_ready'] and not manifest['physical_completeness_claim']
    assert manifest['retention']['partitions']['native_only_retained_entries'] == (primitive == 2)
    assert validate_scene_response_snapshot(manifest, payload)


@pytest.mark.parametrize('start,end,retained', [
    ((-1, .5, .5), (2, .5, .5), True),
    ((2, .5, .5), (-1, .5, .5), True),
    ((.5, .5, .5), (2, .5, .5), True),
    ((-1, -1, -1), (2, 2, 2), True),
    ((-1, 0, .5), (2, 0, .5), True),  # Lower face included.
    ((-1, 1, .5), (2, 1, .5), False),  # Upper face excluded.
    ((-1, -1, .5), (1, -1, .5), False),
    ((-1, 0, .5), (1, -2, .5), False),
    ((-1, 1, .5), (1, -1, .5), False),  # Single corner touch, no interval.
    ((2, .5, .5), (2, .5, .5), False),
    ((2, .5, .5), (3, .5, .5), False),
    ((-1, .5, .5), (0, .5, .5), True),  # Existing endpoint support retained.
    ((2, .5, .5), (1, .5, .5), False),  # Upper-endpoint touch only.
    ((-1e30, .5, .5), (1e30, .5, .5), True),  # Float64 parameters would collapse.
    ((1e30, .5, .5), (-1e30, .5, .5), True),
    ((-1e30, 1e30, .5), (1e30, 1e30, .5), False),
    ((-1, 1, .5), (1, -float(np.nextafter(np.float32(1), np.float32(0))), .5), True),
    ((-1, 1, .5), (1, -float(np.nextafter(np.float32(1), np.float32(2))), .5), False),
])
def test_half_open_finite_segment_scope(start, end, retained):
    s = store()
    s.consume_forward(metadata(), segment(start, end))
    assert s.retained_entries == int(retained)
    assert s.omitted_out_of_scope == int(not retained)


@pytest.mark.parametrize('mutation', ['missing_direction', 'nan_direction', 'nan_origin', 'other_semantic'])
def test_insufficient_segment_evidence_does_not_rescue_an_outside_endpoint(mutation):
    s = store()
    rows = segment()
    if mutation == 'missing_direction': rows[0, 114:117] = 0
    elif mutation == 'nan_direction': rows[0, 114] = 0x7fc12345
    elif mutation == 'nan_origin': rows[0, 117] = 0x7fc12345
    else: rows[0, 0] = 1
    s.consume_forward(metadata(), rows)
    assert s.retained_entries == 0 and s.omitted_out_of_scope == 1


def test_explicit_recipient_scope_still_rejects_crossing_segment():
    s = store()
    s.recipient_namespaces = frozenset({2})
    s.consume_forward(metadata(), segment())
    assert s.retained_entries == 0 and s.omitted_out_of_scope == 1


def test_escape_stays_unlocated_raw_data_and_never_acquires_a_segment():
    s = store()
    rows = segment()
    rows[0, 5] = 0xffffffff
    assert not s._forward_segment_overlaps_bounds(rows[0])
    s.consume_forward(metadata(), rows)
    manifest, payload = s.snapshot()
    index = np.frombuffer(payload, '<u4', count=20)
    assert index[11] == 0 and np.all(index[16:19] == 0xffffffff)
    assert manifest['retention']['partitions']['native_only_retained_entries'] == 1


def test_crossing_native_only_flood_keeps_existing_surface_partition_and_budget():
    s = store()
    s.consume_forward(metadata(), segment(end=(.5, .5, .5)))
    rows = np.repeat(segment(primitive=2), s.capacity*3, axis=0)
    s.consume_forward(metadata(1), rows)
    assert s.evicted_surface_entries == 0
    assert s.retained_entries == 1 + s.native_only_capacity
    assert s.evicted_native_only_entries == len(rows) - s.native_only_capacity
    assert s.native.shape == (s.capacity, 384)
    assert validate_scene_response_snapshot(*s.snapshot())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_roi_cannot_claim_segment_support(bad):
    with pytest.raises(ValueError, match='bounds'):
        store(bounds=((0, 0, 0), (bad, 1, 1)))


def test_retention_agrees_with_independent_plane_event_oracle():
    s = store()
    rng = np.random.default_rng(20261112)
    for _ in range(250):
        start, end = rng.uniform(-2, 3, (2, 3)).astype(np.float32).astype(np.float64)
        delta = end - start
        times = [0., 1.]
        for axis in range(3):
            if delta[axis]:
                times += [(edge-start[axis])/delta[axis] for edge in (0., 1.)
                          if 0 < (edge-start[axis])/delta[axis] < 1]
        times = sorted(times)
        expected = any(np.all((p >= 0) & (p < 1)) for p in
            (start + delta*((a+b)/2) for a, b in zip(times, times[1:]) if b > a))
        assert s._forward_segment_overlaps_bounds(segment(start, end)[0]) == expected
