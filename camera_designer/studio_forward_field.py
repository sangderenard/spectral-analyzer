"""Opt-in immutable copies of the existing forward estimator, not a new estimator."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

import numpy as np


_ARTIFACT = re.compile(r'light-field-[0-9a-f]{32}\.bin')
_INCOMPLETE = re.compile(r'light-field-[0-9a-f]{32}\.tmp')


def _regular_file(path):
    """Never adopt or remove symlinks or Windows reparse points."""
    info = path.lstat()
    return (stat.S_ISREG(info.st_mode) and
            not getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024))


def validate_receive_light_field(config):
    from camera_designer.studio_directional_field import recording_options
    recording_options(config.get('light_field_estimator','track-length-throughput-v1'),config.get('light_field_grid_size'))
    if 'light_field_stencil' in config:
        from camera_designer.studio_directional_field import validate_stencil
        validate_stencil(config['light_field_stencil'])
    internal=config.get('light_field_internal_media',False)
    if type(internal) is not bool:raise ValueError('light_field_internal_media must be a boolean')
    if internal:
        from camera_designer.studio_native_emitters import sampling_settings
        if not sampling_settings(config)['refresh_each_sweep']:raise ValueError('internal directional recording requires refreshed v6 populations')
    enabled = config.get('receive_light_field', False)
    if type(enabled) is not bool:
        raise ValueError('receive_light_field must be a boolean')
    if enabled and config.get('transport', 'fixed:1') != 'fixed:3':
        raise ValueError('receive_light_field requires fixed:3 transport')
    information=config.get('light_field_information')
    if information is not None:
        if information!='evaluated-responses-v1' or not enabled:
            raise ValueError('evaluated response copies require explicit field information opt-in')
    return enabled


def field_to_linear_srgb(counts, dimensions):
    # The producer stores direct 650/550/450 nm energy, not film measurements.
    # Reuse the existing reference colorimetry WITHOUT its sensor inversion.
    from camera_designer.marble_color_profile import XYZ_TO_SRGB, _LINE_TO_XYZ
    raw = np.asarray(counts)
    if raw.shape != (3, *reversed(dimensions)) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('forward field must contain finite nonnegative fixed3 counts in band/z/y/x order')
    lines = np.moveaxis(raw, 0, -1).astype(np.float64) / 65535.0
    result = np.ascontiguousarray(lines @ (XYZ_TO_SRGB @ _LINE_TO_XYZ).T, dtype='<f4')
    if not np.isfinite(result).all():
        raise ValueError('forward field exceeds finite float32 range')
    return result


class ForwardFieldPublisher:
    """At most one readback per second; keep four owned immutable artifacts.

    Invalidation removes metadata immediately but never resets the estimator.
    The latest and status-advertised artifacts survive pruning. Reconcile only
    exact artifact names in this job directory, including after worker restart.
    """
    def __init__(self, output, digest, *, interval=1.0):
        self.output = Path(output)
        self.digest = digest
        self.interval = max(1.0, float(interval))
        self.sequence = 0
        self.latest = None
        self.last_readback = float('-inf')
        existing = []
        self.incomplete = deque()
        for path in self.output.iterdir():
            if not (_ARTIFACT.fullmatch(path.name) or _INCOMPLETE.fullmatch(path.name)):
                continue
            try:
                if not _regular_file(path):
                    continue
                if _INCOMPLETE.fullmatch(path.name):
                    self.incomplete.append(path)
                else:
                    existing.append((path.lstat().st_mtime_ns, path.name, path))
            except FileNotFoundError:
                continue
        self.files = deque(item[2] for item in sorted(existing))

    def invalidate(self):
        self.latest = None

    def _prune(self):
        # A previous worker may have stopped between writing and atomic rename.
        # Locked leftovers backpressure publication instead of accumulating.
        while self.incomplete:
            path = self.incomplete[0]
            try:
                if _regular_file(path):
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False
            self.incomplete.popleft()
        protected = {self.files[-1].name} if self.files else set()
        if self.latest is not None:
            protected.add(self.latest['name'])
        status_path = self.output / 'status.json'
        try:
            if _regular_file(status_path):
                status = json.loads(status_path.read_text(encoding='utf-8'))
                advertised = status.get('light_field', {}).get('name')
                if isinstance(advertised, str) and _ARTIFACT.fullmatch(advertised):
                    protected.add(advertised)
        except (OSError, ValueError, AttributeError):
            pass
        while len(self.files) >= 4:
            path = next((p for p in self.files if p.name not in protected), None)
            if path is None:
                return False
            try:
                if _regular_file(path):
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False
            self.files.remove(path)
        return True

    def snapshot(self, renderer, field, *, live_revision=0, sweep=0,
                 sensor_samples=0, ray_jobs=0, accumulation='additive since last clear',
                 scene_ack=None, now=None):
        progressive=(field.get('directional_record') or {}).get('progressive')
        if progressive and self.latest is not None and self.latest.get('live_revision')==int(live_revision):
            previous=self.latest.get('directional',{}).get('source_populations')
            if previous==progressive['metadata'] and self.latest.get('scene_sha256')==field['directional_record'].get('scene_sha256'):
                return self.latest
        supplied_now = now
        now = time.monotonic() if now is None else now
        if now - self.last_readback < self.interval:
            return self.latest
        # Do not grow disk use without bound if Windows readers hold old files.
        if not self._prune():
            return self.latest
        from OpenGL.GL import glMemoryBarrier, GL_TEXTURE_UPDATE_BARRIER_BIT
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT)
        dimensions = tuple(int(d) for d in field['dims'])
        if len(dimensions) != 3 or any(d <= 0 for d in dimensions):
            raise ValueError('invalid forward field dimensions')
        bmin = np.asarray(field['bmin'], dtype=float)
        bmax = np.asarray(field['bmax'], dtype=float)
        if (bmin.shape != (3,) or bmax.shape != (3,) or not np.isfinite(bmin).all()
                or not np.isfinite(bmax).all() or not np.all(bmax > bmin)):
            raise ValueError('invalid forward field bounds')
        counts = renderer._gpu_readback_prebuilt(field['tex_bands'], dimensions)
        rgb = field_to_linear_srgb(counts, dimensions)
        payload = rgb.tobytes(order='C')
        payload_parts = [payload]
        sections = {}
        directional_unavailable = None
        if field.get('directional_record') is not None:
            from camera_designer.studio_directional_field import bundle_sections, DirectionalUnavailable
            try:
                extra, sections = bundle_sections(field['directional_record'], field['source_buf'], len(payload), streaming=True)
                payload_parts.extend(extra)
            except DirectionalUnavailable as unavailable:
                # Optional recording must never kill the primary exposure or
                # silently reset/normalize the existing scalar estimator.
                directional_unavailable = {
                    'reason': unavailable.reason,
                    'recovery': unavailable.recovery,
                    'fallback': 'unchanged canonical scalar v1; no directional approximation published',
                }
        name = f'light-field-{uuid.uuid4().hex}.bin'
        target = self.output / name
        temporary = target.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                for part in payload_parts:
                    stream.write(part)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        payload_length=sum(len(part) for part in payload_parts)
        payload_hash=hashlib.sha256()
        for part in payload_parts:payload_hash.update(part)
        self.sequence += 1
        self.latest = {
            'schema': 'studio.forward-field.v1', 'digest': self.digest,
            'live_revision': int(live_revision), 'sequence': self.sequence,
            'sweep': int(sweep), 'sensor_samples': int(sensor_samples),
            'name': name, 'byte_length': payload_length,
            'sha256': payload_hash.hexdigest(),
            'dimensions': list(dimensions),
            'bounds_min_m': bmin.tolist(),
            'bounds_max_m': bmax.tolist(),
            'format': 'rgb-f32le', 'layout': 'zyx-interleaved-rgb',
            'channels': ['R', 'G', 'B'], 'color_space': 'linear-srgb',
            'transport_wavelengths_nm': [650, 550, 450],
            'profile': 'marble-fixed3-cie1931-d65-v1',
            'quantization_divisor': 65535, 'ray_jobs': int(ray_jobs),
            'accumulation': accumulation,
            'normalization': 'none; not divided by ray count or sensor samples',
            'limits': 'Three-line reference colorimetry; signed out-of-gamut RGB retained. Scalar voxel energy, not directional radiance or calibrated irradiance. No film response, tone mapping, gamma, denoising or temporal averaging.',
        }
        if field.get('surface_transport') is not None:self.latest['surface_transport']=field['surface_transport']
        if directional_unavailable is not None:
            self.latest['directional_unavailable'] = directional_unavailable
        if sections:
            self.latest.update(sections)
            self.latest['schema'] = ('studio.forward-field.v6' if sections['directional'].get('source_populations') else 'studio.forward-field.v5' if sections['directional'].get('region') else 'studio.forward-field.v4' if sections['directional']['estimator']=='track-length-root-sweep-v1' else 'studio.forward-field.v3')
            if field.get('geometry_sha256') is not None:
                self.latest['geometry_sha256'] = field['geometry_sha256']
            self.latest['limits'] += (' Directional section records actual interior/exterior Beer-integrated track-length throughput density in explicit coarse angular bins.' if sections['directional'].get('internal_media') else ' Directional section records exterior segment track-length throughput density in explicit coarse angular bins.') + ' This is not a complete angular radiance function. Signed 32-bit sums require bounded fresh sweeps; no individual path ancestry. Material/triangle IDs use the assembled server scene, not browser mesh indices.'
        if sections.get('directional',{}).get('source_populations'):
            self.latest['limits'] += ' Directional values are a mean of complete refreshed source populations; deterministic population provenance is retained, not individual ray ancestry. Scalar remains fresh per sweep.'
        if scene_ack is not None:
            self.latest['scene_ack'] = dict(scene_ack)
        self.last_readback = time.monotonic() if supplied_now is None else now
        self.files.append(target)
        return self.latest
