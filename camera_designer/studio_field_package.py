"""CPU-only immutable v7 package/container contract shared by publisher and API.

This validates transport completeness and typed observations, not convergence,
darkness, visibility, or the completion of a screen-space consumer.
"""
import hashlib
import json
import re
import struct
import os
import time
import stat
from pathlib import Path
from contextlib import contextmanager

SCHEMA = 'studio.forward-field.v7'
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_CHUNK_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PACKAGES = 3
MAX_LEASES = 8
LEASE_SECONDS = 600
HEX = re.compile(r'^[0-9a-f]{32}$')
SHA = re.compile(r'^[0-9a-f]{64}$')
CHUNK = re.compile(r'^light-field-([0-9a-f]{32})-([0-9]{4})\.bin$')
SECTION_CONTRACTS = (
    ('base', 1, 'studio.forward-field.v6'),
    ('arrivals', 124, 'arrival-domain-id-record-v1'),
    ('domains', 132, 'domain-key-record-v1'),
    ('scene_contract', 1, 'native-scene-domains-v1'),
)


def require(condition, message='invalid v7 field package'):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=2**53-1):
    return type(value) is int and low <= value <= high


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def chunk_name(package_id, index):
    require(isinstance(package_id, str) and bool(HEX.fullmatch(package_id)) and integer(index, 0, 9999))
    return f'light-field-{package_id}-{index:04d}.bin'


def manifest_name(package_id):
    require(isinstance(package_id, str) and bool(HEX.fullmatch(package_id)))
    return f'light-field-{package_id}.manifest.json'


def lease_name(package_id):
    require(isinstance(package_id, str) and bool(HEX.fullmatch(package_id)))
    return f'light-field-{package_id}.leases.json'


@contextmanager
def package_lock(directory, package_id=None, timeout=5.0):
    """One bounded cross-process job lock for API leases and publisher pruning.

    The single lockfile is never unlinked, avoiding a second inode/handle lock
    domain while another process is waiting. No per-population lockfile growth.
    """
    path = Path(directory) / 'light-field-packages.lock'
    if path.exists():
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
                and not getattr(info, 'st_file_attributes', 0) & 1024, 'unsafe package lockfile')
    stream = path.open('a+b')
    locked = False
    try:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b'\0'); stream.flush()
        until = time.monotonic() + timeout
        while not locked:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.monotonic() >= until:
                    raise ValueError('field package lock is busy') from None
                time.sleep(.01)
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def validate_manifest(manifest):
    """Return section map after bounded, exact membership/span validation."""
    require(isinstance(manifest, dict) and manifest.get('schema') == SCHEMA)
    package_id = manifest.get('package_id')
    require(isinstance(package_id, str) and bool(HEX.fullmatch(package_id)))
    for key in ('digest', 'scene_sha256', 'config_sha256'):
        require(isinstance(manifest.get(key), str) and bool(SHA.fullmatch(manifest[key])))
    require(integer(manifest.get('live_revision')) and integer(manifest.get('sequence')))
    require(integer(manifest.get('byte_length'), 1, MAX_TOTAL_BYTES), 'complete v7 package exceeds 512 MiB admission')
    sections = manifest.get('sections')
    require(isinstance(sections, list) and len(sections) == len(SECTION_CONTRACTS))
    by_id = {}; offset = 0
    for section, (name, stride, fmt) in zip(sections, SECTION_CONTRACTS):
        require(isinstance(section, dict) and section.get('id') == name
                and type(section.get('stride_bytes')) is int and section['stride_bytes'] == stride
                and section.get('format') == fmt and isinstance(section.get('metadata'), dict))
        require(integer(section.get('count')) and integer(section.get('byte_length'), 0, MAX_TOTAL_BYTES)
                and section['byte_length'] == section['count'] * stride
                and type(section.get('offset_bytes')) is int and section['offset_bytes'] == offset)
        offset += section['byte_length']; by_id[name] = section
    require(offset == manifest['byte_length'])
    base = by_id['base']['metadata']
    require(base.get('schema') == 'studio.forward-field.v6' and base.get('digest') == manifest['digest']
            and base.get('live_revision') == manifest['live_revision'] and type(base.get('live_revision')) is int
            and base.get('scene_sha256') == manifest['scene_sha256']
            and base.get('byte_length') == by_id['base']['byte_length'])
    arrivals = by_id['arrivals']['metadata']
    require(integer(arrivals.get('index_count'), 1, 64**3 * 12))
    require(arrivals.get('wavelengths_nm') == [650, 550, 450]
            and arrivals.get('spectral_channels') == ['energy', 'moment_x', 'moment_y', 'moment_z'])
    chunks = manifest.get('chunks')
    require(isinstance(chunks, list) and 1 <= len(chunks) <= 40)
    expected_chunks = []
    for section in sections:
        stride = section['stride_bytes']; capacity = (MAX_CHUNK_BYTES // stride) * stride
        for local in range(0, section['byte_length'], capacity):
            size = min(capacity, section['byte_length'] - local)
            expected_chunks.append((section['id'], local, size, local // stride, size // stride))
    require(len(chunks) == len(expected_chunks))
    for index, (chunk, expected) in enumerate(zip(chunks, expected_chunks)):
        section, local, size, first, count = expected
        require(isinstance(chunk, dict) and chunk.get('name') == chunk_name(package_id, index)
                and chunk.get('section') == section)
        for key, value in (('offset_bytes', local), ('byte_length', size), ('first_record', first), ('record_count', count)):
            require(type(chunk.get(key)) is int and chunk[key] == value)
        require(isinstance(chunk.get('sha256'), str) and bool(SHA.fullmatch(chunk['sha256'])))
    require(len(json.dumps(manifest, allow_nan=False).encode('utf-8')) <= MAX_MANIFEST_BYTES)
    return by_id


def validate_chunk_bytes(manifest, chunk, data):
    """Validate a bounded immutable member, including actual arrival evidence.

    Full native scene semantics remain the canonical scene-contract validator's
    responsibility. Domain IDs are row identities in the complete domain table.
    """
    sections = validate_manifest(manifest)
    require(chunk in manifest['chunks'] and len(data) == chunk['byte_length'])
    require(hashlib.sha256(data).hexdigest() == chunk['sha256'], 'v7 chunk hash mismatch')
    if chunk['section'] == 'arrivals':
        import numpy as np
        rows = np.frombuffer(data, dtype='<u4').reshape(-1, 31)
        spectral = rows[:, 1:13].copy().view('<f4').reshape(-1, 3, 4)
        visits, zero, excluded = rows[:, 13:16], rows[:, 16:19], rows[:, 19:22]
        require(np.isfinite(spectral).all() and np.all(spectral[:, :, 0] >= 0), 'invalid v7 spectral observations')
        require(np.all(rows[:, 0] < sections['arrivals']['metadata']['index_count'])
                and np.all(rows[:, 30] < sections['domains']['count']), 'invalid v7 observation identity')
        require(np.all(zero <= visits) and np.all((spectral[:, :, 0] == 0) | (visits > 0))
                and np.all(np.any((visits > 0) | (excluded > 0), axis=1)), 'invalid v7 arrival counters')
        keys = rows[:, [0, 30]].astype(np.uint64)
        if len(keys) > 1:
            require(np.all((keys[1:, 0] > keys[:-1, 0]) |
                           ((keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] > keys[:-1, 1]))),
                    'v7 arrival keys must be sorted and unique')
    elif chunk['section'] == 'domains':
        # Bounds for actual medium IDs and region membership need scene_contract;
        # the transport layer must not invent domains from a native PBR index.
        require(len(data) % 132 == 0)
    return True
