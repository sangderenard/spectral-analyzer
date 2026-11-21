"""Transactional native host for AbstractUI island-door events.

An explicit registry is the authority boundary. href remains descriptive and
is never fetched or executed. Factories build existing geometry/workspaces;
this module owns no renderer, credentials, service, or alternate map schema.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Callable, Mapping
from native_doorways import DoorCrossing, DoorRuntime


@dataclass
class NativeIsland:
    geometry: Any
    workspace: Any
    runtime: DoorRuntime | None = None
    # Host retains GL objects/menus per island, preserving edits and controls.
    host_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.runtime is None:
            self.runtime = self.geometry.door_runtime()

    @property
    def identity(self):
        return self.geometry.model['identity']


@dataclass(frozen=True)
class PreparedIsland:
    previous: NativeIsland
    destination: NativeIsland
    arrival: DoorCrossing
    event: Mapping[str, Any]
    key: tuple[str, Any]
    generation: int


class NativeIslandLoader:
    """Prepare geometry/arrival before replacing any active host state.

    Call prepare(event), construct/validate target scene on the host, then
    commit(prepared). Failed loads leave current/config/session unchanged.
    Cached workspaces are retained when leaving and returning; switching is
    not re-reading scene.yaml or resetting user edits.
    """
    def __init__(self, initial: NativeIsland, *, initial_cache_key=None):
        self.current = initial
        self._factories: dict[str, Callable[[Mapping[str, Any]], NativeIsland]] = {}
        self._cache = {(initial.identity, initial_cache_key): initial}
        self._key_resolvers = {}
        self._generation = 0

    def register(self, island: str, factory: Callable[[Mapping[str, Any]], NativeIsland], *, cache_key=None):
        if not island or not callable(factory):
            raise ValueError('named island factory required')
        if island in self._factories:
            raise ValueError('island already registered')
        self._factories[island] = factory
        if cache_key is not None:
            if not callable(cache_key): raise ValueError("cache key resolver must be callable")
            self._key_resolvers[island] = cache_key

    def _validated_event(self, event):
        source = next((o for o in self.current.runtime.openings if o['identity'] == event.get('source')), None)
        if (source is None or source.get('port_role') != 'in' or source.get('transition') != 'door'
                or not source.get('island') or not self.current.runtime._enabled(source)):
            raise ValueError('event is not a current enabled island door')
        for field in ('island', 'pair', 'href'):
            if source.get(field) != event.get(field):
                raise ValueError(f'event does not match declared {field}')
        result = {key: source.get(key) for key in ('island','pair','href')}
        result['source'] = source['identity']
        declared = source.get('treadmill')
        if declared:
            key = source.get('coincident_with', source['identity'])
            expected = {**declared,'offset':self.current.runtime.treadmill_offsets.get(key,declared['offset'])}
            if event.get('treadmill') != expected:
                raise ValueError('event does not match current treadmill window')
            result['treadmill'] = expected
        elif event.get('treadmill') is not None:
            raise ValueError('undeclared treadmill')
        return result

    def prepare(self, event: Mapping[str, Any], *, actor='player', radius=.15, now_ms: float) -> PreparedIsland:
        event = self._validated_event(event)
        identity = event['island']
        offset = event.get('treadmill',{}).get('offset')
        resolver = self._key_resolvers.get(identity)
        key = (identity, resolver(event) if resolver is not None else offset)
        destination = self._cache.get(key)
        if destination is None:
            factory = self._factories.get(identity)
            if factory is None:
                raise KeyError(f'island is not registered: {identity}')
            destination = factory(deepcopy(event))
            if not isinstance(destination, NativeIsland) or destination.identity != identity:
                raise ValueError('factory returned a different island')
        # Preview arrival on separate runtime: a failed host attachment must
        # not mutate even a cached destination's cooldowns or selector state.
        runtime = destination.runtime
        preview = DoorRuntime(runtime.openings, runtime.geometry, runtime.cooldown_ms,
                              dict(runtime.dial_positions), dict(runtime.treadmill_offsets), dict(runtime.cooldowns))
        arrival = preview.arrive(actor, event['pair'], radius=radius, now_ms=now_ms)
        return PreparedIsland(self.current, destination, arrival, event, key, self._generation)

    def commit(self, prepared: PreparedIsland, *, actor='player') -> NativeIsland:
        if prepared.previous is not self.current or prepared.generation != self._generation:
            raise ValueError('stale island transition')
        self._cache[prepared.key] = prepared.destination
        self.current = prepared.destination
        self.current.runtime.cooldowns[actor] = prepared.arrival.cooldown_until_ms
        self._generation += 1
        return self.current


def document_island(model: Mapping[str, Any], workspace_factory: Callable, *, scale=1.0) -> NativeIsland:
    """Build an already-authorized Living Map document, no runtime compilation.

    workspace_factory(geometry) uses the host's existing workspace/config
    factory. All control capabilities and account sessions stay with that host.
    """
    from src.compiler.abstract_ui_viewports import document_geometry
    from nodus_world_geometry import from_document_geometry
    if model.get('schema') != 'abstract-ui-div-map-v0':
        raise ValueError('expected existing Living Map document schema')
    geometry = from_document_geometry(model, document_geometry(model)).scaled(scale)
    return NativeIsland(geometry, workspace_factory(geometry))

def read_local_document(path):
    """Read only an explicitly supplied local JSON map; never follow href."""
    import json
    from pathlib import Path
    text = str(path)
    if '://' in text or text.startswith(('\\\\', '//')):
        raise ValueError('island documents must be local files, not network locations')
    with Path(path).open(encoding='utf-8-sig') as source:
        model = json.load(source)
    if not isinstance(model, dict) or model.get('schema') != 'abstract-ui-div-map-v0' or not model.get('identity'):
        raise ValueError('expected named abstract-ui-div-map-v0 document')
    return model


def read_island_manifest(path):
    """Resolve relative local files against the explicit manifest directory."""
    import json
    from pathlib import Path
    text = str(path)
    if '://' in text or text.startswith(('\\\\', '//')):
        raise ValueError('island manifest must be local')
    path = Path(path)
    with path.open(encoding='utf-8-sig') as source:
        entries = json.load(source)
    if not isinstance(entries, dict):
        raise ValueError('island manifest must map identities to local filenames')
    documents = {}
    for identity, filename in entries.items():
        if not isinstance(identity, str) or not identity or not isinstance(filename, str):
            raise ValueError('manifest identities and filenames must be strings')
        if '://' in filename or filename.startswith(('\\\\', '//')):
            raise ValueError('island manifest cannot fetch network documents')
        target = Path(filename)
        if not target.is_absolute():
            target = path.parent / target
        model = read_local_document(target)
        if model['identity'] != identity:
            raise ValueError('manifest identity does not match document')
        documents[identity] = model
    return documents
