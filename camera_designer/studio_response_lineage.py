"""Opt-in copies of identities at native packing/filter/compaction events.

No geometry export, positional matching, intersection, or optical evaluation.
Zero namespace has no browser recipient. Source ordinals are the original
MRP1 subject or fixture-array ordinals, never native material IDs.
"""
import numpy as np
from copy import deepcopy

VERSION = 'studio-response-lineage-v1'
ROW_CHANNELS = ['source_namespace', 'source_face', 'primitive_kind', 'primitive_index']
SPHERE_CHANNELS = ['sphere_index', 'native_owner_triangle', 'source_start', 'source_count']
RECIPIENT_INDEX_LIMIT = 0x0fffffff


class NativeResponseLineage:
    def __init__(self, camera_triangle_count):
        self.rows = np.full((int(camera_triangle_count), 4), -1, dtype='<i4')
        self.rows[:, 0] = 0  # Native-only housing is not a Phong receiver.
        self.rows[:, 2] = 0
        self.spheres = []
        self.sources = []

    def append(self, namespace, source_id, native_offset, retained):
        """Called with the actual packer's np.flatnonzero(valid), not rebuilt."""
        if namespace not in (1, 2) or not isinstance(source_id, str) or not source_id:
            raise ValueError('explicit subject or fixture source identity required')
        selected = np.asarray(retained)
        if (native_offset != len(self.rows) or selected.ndim != 1 or
                np.any(selected < 0) or np.any(selected != np.floor(selected)) or
                np.any(selected >= RECIPIENT_INDEX_LIMIT) or
                (len(selected) > 1 and np.any(np.diff(selected) <= 0))):
            raise ValueError('native identity copy does not match original packing order')
        rows = np.full((len(selected), 4), -1, dtype='<i4')
        rows[:, 0] = namespace
        rows[:, 1] = selected
        rows[:, 2] = 0
        self.rows = np.concatenate((self.rows, rows))
        self.sources.append(dict(namespace=namespace, source_id=source_id))

    def compact_spheres(self, keep, old_to_new, representatives, spheres):
        """Copy the exact existing selection and remapping before discard."""
        if len(keep) != len(self.rows) or len(representatives) != len(spheres):
            raise ValueError('sphere selection and lineage lengths disagree')
        rows = self.rows[np.asarray(keep, bool)].copy()
        copied = []
        for si, (old, sphere) in enumerate(zip(representatives, spheres)):
            owner = int(old_to_new[old])
            if rows[owner, 0] != 1:
                raise ValueError('exact subject sphere lost its submitted source identity')
            rows[owner, 2:] = [1, si]
            copied.append([si, owner, int(sphere['start']), int(sphere['count'])])
        self.rows = rows
        self.spheres = copied

    def snapshot(self, *, geometry_sha256=None, revision=None, fixture_parts=None):
        return dict(version=VERSION, row_channels=list(ROW_CHANNELS),
            rows=self.rows.copy(), sphere_channels=list(SPHERE_CHANNELS),
            spheres=np.asarray(self.spheres, dtype='<i4').reshape(-1, 4),
            namespaces={'unmatched_native': 0, 'subject': 1, 'fixtures': 2, 'exact_sphere': 3},
            sources=[dict(value) for value in self.sources],
            geometry_sha256=geometry_sha256, revision=revision,
            fixture_parts=deepcopy(fixture_parts),
            virtual_partition='no mesh recipient; retain copied event kind and native owner separately',
            missing_phong_surface='unmatched recipient, alpha zero; never map by material or position')


def copied_recipient(lineage, native_hit, primitive_kind=0, primitive_owner=-1):
    """Server packaging lookup of copied identities, never a ray query.

    Returns (namespace, source ordinal or sphere index). Primitive kind 2 is
    a virtual partition, not the host face whose record it borrows. Missing
    and camera-only recipients remain (0,-1). Visibility is a separate fact.
    """
    if primitive_kind not in (0, 1, 2):
        raise ValueError('unknown copied primitive kind')
    rows = lineage['rows']
    if primitive_kind == 2 or not 0 <= native_hit < len(rows):
        return 0, -1
    namespace, source_face, kind, owner = map(int, rows[native_hit])
    if primitive_kind == 1:
        if kind != 1 or owner != primitive_owner:
            raise ValueError('copied sphere event disagrees with its original owner identity')
        return 3, owner  # Explicit sphere primitive, not a guessed facet.
    if kind != 0:
        raise ValueError('native primitive kind is missing or disagrees with lineage')
    return (namespace, source_face) if namespace else (0, -1)


def recipient_id(namespace, index):
    """Exact integer encoding also used by browser flat attribute 7."""
    if namespace == 0:
        return 0
    if namespace not in (1, 2, 3) or int(index) != index or not 0 <= index < RECIPIENT_INDEX_LIMIT:
        raise ValueError('invalid copied response recipient identity')
    return (int(namespace) << 28) | (int(index) + 1)
