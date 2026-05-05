"""Canonical controls + actions hierarchy and dispatcher.

This is the single, exclusive module through which any control anywhere in
the application is expressed and through which any action in the application
is registered and executed.

Design summary
--------------
- Every control is a ``KnobSpec`` (single value, possibly grouped) or a
  ``Panel`` (a named bag of knobs / sub-panels). No ad-hoc tuple controls
  are accepted anywhere; producers must convert to KnobSpecs first.
- Controls live on a hierarchy graph::

      Global
       |
       +-- Room (RoomControl owner)
       |    |
       |    +-- Object (PlacedObject / DutyStation / Camera / ...)
       |         |
       |         +-- StationControls   (panels + knobs inside the object)
       |         +-- TriangleGroupActions  (mesh-triangle actions on object)
       |
       +-- Object (when owned directly by Global, not a room)

  ``StationControls`` and ``TriangleGroupActions`` are *peers* under the
  owning object node. Stations do not exist as a separate level above
  objects; they live inside the object that owns them.

- Each leaf that can fire an action carries an ``action_key``. Action keys
  are looked up in the singleton ``ActionRegistry`` to resolve a callable
  (or a script-path/hook-name pair) plus metadata.

- The pygame main loop (and any other event source) consumes input cheaply:
  on each consumed event it calls :func:`enqueue_action`, which hands the
  ``(action_key, context)`` tuple to a dedicated worker thread. The worker
  thread runs the registered hook so neither input collection nor the
  OpenGL context tick is blocked by user-action code.

- The OpenGL shader dispatcher and the event reader both live in the main
  loop (this module deliberately does not own OpenGL state). This module
  only exposes the registry, the hierarchy graph, and the action queue.

Public API
----------
- :class:`KnobSpec` (resolved from ``signal_generator_v2`` if available).
- :class:`Panel`.
- :class:`OwnerScope`, :class:`ControlNode`, :class:`ControlGraph`.
- :class:`ActionBinding`, :class:`ActionRegistry`,
  :func:`get_action_registry`.
- :class:`ActionDispatcher`, :func:`get_dispatcher`,
  :func:`enqueue_action`, :func:`start_action_dispatcher`,
  :func:`stop_action_dispatcher`.
- :func:`slider_defs_to_knobs`, :func:`knobs_to_object_nodes`.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# KnobSpec resolution
# ---------------------------------------------------------------------------

def _resolve_knob_spec() -> Any:
    try:
        from signal_generator_v2 import KnobSpec  # type: ignore
        return KnobSpec
    except Exception:
        @dataclass
        class KnobSpec:  # type: ignore[no-redef]
            name: str
            label: str
            dtype: str = "float"
            default: Any = None
            low: float = 0.0
            high: float = 1.0
            step: float = 0.0
            unit: str = ""
            choices: list[str] | None = None
            is_log: bool = False
            group: str = ""
            fmt: str = ".3g"
            source_class: str = ""
            rebuild_layout: bool = False
            visible_when: tuple[str, str] | None = None

        return KnobSpec


KnobSpec = _resolve_knob_spec()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Panel:
    """A named bundle of knobs and/or nested panels."""

    name: str
    label: str = ""
    knobs: list[Any] = field(default_factory=list)
    panels: list["Panel"] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Control hierarchy
# ---------------------------------------------------------------------------

class OwnerScope(str, Enum):
    """Where a node sits in the hierarchy.

    The two real "owners" of subordinate state are GLOBAL and OBJECT (and
    ROOM, which exists only as an organizational parent for objects).
    Every owner has the same five subordinate peer axes, so the meaning
    of each subtree is unambiguous:

    - CONTROLS:  KnobSpec / Panel descriptors. Never any shader code.
    - ACTIONS:   Triangle-group action sites. Never any shader code.
    - SHADERS:   Custom shader nodes (with cadence + flip buffers).
                 Never any control or action descriptor.
    - INPUTS:    Placeholder for FIFO edge endpoints feeding *into* this
                 owner from a future networked compute subsystem. Stores
                 only node ids; no transport logic lives here.
    - OUTPUTS:   Placeholder for FIFO edge endpoints emitted *out* of this
                 owner to a future networked compute subsystem. Stores
                 only node ids; no transport logic lives here.
    """

    GLOBAL = "global"
    ROOM = "room"
    OBJECT = "object"
    CONTROLS = "controls"
    ACTIONS = "actions"
    SHADERS = "shaders"
    INPUTS = "inputs"
    OUTPUTS = "outputs"


class PortClass(str, Enum):
    """Physical contact geometry of a FIFO endpoint.

    Cables refuse to plug across classes, so naive and advanced wiring
    cannot accidentally bridge.

    - ``NAIVE_PRONG``: the default.  Male prong contact, naive layer.
    - ``ADVANCED_FLUSH``: flush-mount female bore with cover plate;
      mates only with other flush plates.
    """

    NAIVE_PRONG = "naive_prong"
    ADVANCED_FLUSH = "advanced_flush"


class PortRole(str, Enum):
    """Logical role a port plays inside its star.

    Orthogonal to :class:`PortClass` (a WAN port is still a naive
    prong physically).

    - ``STANDARD``: ordinary star-member port.  Star tick delivers
      locally-addressed envelopes here and drains its outputs.
    - ``WAN_BOUNDARY``: marks the port as the boundary of *this* star.
      Envelopes routed across this port carry loop-protection markers;
      another star treats inbound traffic at its WAN port as ordinary
      ingress.
    - ``GATEWAY``: reserved for the singleton gateway object.  Refuses
      to join any star and runs synchronously on the controller thread.
    """

    STANDARD = "standard"
    WAN_BOUNDARY = "wan_boundary"
    GATEWAY = "gateway"


def port_class_of(node: "ControlNode | None") -> "PortClass":
    """Return the :class:`PortClass` recorded on an endpoint node.

    Defaults to :attr:`PortClass.NAIVE_PRONG` when no class has been
    annotated (which is the case for every legacy endpoint registered
    before the port-class system existed).
    """
    if node is None:
        return PortClass.NAIVE_PRONG
    raw = node.payload.get("port_class")
    if isinstance(raw, PortClass):
        return raw
    if isinstance(raw, str):
        try:
            return PortClass(raw)
        except ValueError:
            return PortClass.NAIVE_PRONG
    return PortClass.NAIVE_PRONG


def port_role_of(node: "ControlNode | None") -> "PortRole":
    """Return the :class:`PortRole` recorded on an endpoint node.

    Defaults to :attr:`PortRole.STANDARD`.
    """
    if node is None:
        return PortRole.STANDARD
    raw = node.payload.get("port_role")
    if isinstance(raw, PortRole):
        return raw
    if isinstance(raw, str):
        try:
            return PortRole(raw)
        except ValueError:
            return PortRole.STANDARD
    return PortRole.STANDARD


@dataclass(slots=True)
class ControlNode:
    """A node in the control graph.

    A node may carry a ``knob`` (leaf control), a ``panel`` (named bundle),
    a ``triangle_group`` name (mesh-triangle action site), or an
    ``action_key`` (link into the action registry). Children represent
    nested ownership in the hierarchy.
    """

    key: str
    label: str = ""
    scope: OwnerScope = OwnerScope.OBJECT
    owner_id: str = ""
    knob: Any | None = None
    panel: Panel | None = None
    triangle_group: str = ""
    action_key: str = ""
    control_action: str = ""
    children: list["ControlNode"] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def add(self, child: "ControlNode") -> "ControlNode":
        self.children.append(child)
        return child

    def walk(self) -> Iterable["ControlNode"]:
        yield self
        for ch in self.children:
            yield from ch.walk()


class ControlGraph:
    """Hierarchical container for all ``ControlNode`` instances.

    Layout is strict and unambiguous::

        global  (OwnerScope.GLOBAL)
          ├─ controls/   (CONTROLS)
          ├─ actions/    (ACTIONS)
          ├─ shaders/    (SHADERS)
          ├─ inputs/     (INPUTS)   — FIFO edge endpoints (placeholder)
          ├─ outputs/    (OUTPUTS)  — FIFO edge endpoints (placeholder)
          └─ room/<id>/  (ROOM)
               └─ object/<id>/  (OBJECT)
                    ├─ controls/   (CONTROLS)
                    ├─ actions/    (ACTIONS)
                    ├─ shaders/    (SHADERS)
                    ├─ inputs/     (INPUTS)
                    └─ outputs/    (OUTPUTS)

    Objects may also attach directly under GLOBAL when not in a room.
    Every subordinate axis is a peer of the others; they never own each
    other and they never live above their owner.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.root = ControlNode(
            key="global",
            label="Global",
            scope=OwnerScope.GLOBAL,
            owner_id="global",
        )
        self._index_by_key: dict[str, ControlNode] = {self.root.key: self.root}
        self._index_by_action: dict[str, ControlNode] = {}
        self._index_by_triangle_group: dict[str, ControlNode] = {}
        self._index_by_shader: dict[str, ControlNode] = {}
        # Attach the global owner's three subordinate axes.
        self._attach_owner_axes(self.root, owner_id="global")

    # ---- attachment helpers ----

    def _attach_owner_axes(self, owner: ControlNode, *, owner_id: str) -> None:
        for scope, label in (
            (OwnerScope.CONTROLS, "Controls"),
            (OwnerScope.ACTIONS, "Actions"),
            (OwnerScope.SHADERS, "Shaders"),
            (OwnerScope.INPUTS, "Inputs"),
            (OwnerScope.OUTPUTS, "Outputs"),
        ):
            node = ControlNode(
                key=f"{owner.key}/{scope.value}",
                label=label,
                scope=scope,
                owner_id=owner_id,
            )
            owner.add(node)
            self._index_by_key[node.key] = node

    def attach_room(self, room_id: str, label: str = "") -> ControlNode:
        with self._lock:
            existing = self._index_by_key.get(f"room/{room_id}")
            if existing is not None:
                return existing
            node = ControlNode(
                key=f"room/{room_id}",
                label=label or room_id,
                scope=OwnerScope.ROOM,
                owner_id=room_id,
            )
            self.root.add(node)
            self._index_by_key[node.key] = node
            return node

    def attach_object(
        self,
        object_id: str,
        *,
        room_id: str = "",
        label: str = "",
    ) -> ControlNode:
        with self._lock:
            obj_key = f"object/{object_id}"
            existing = self._index_by_key.get(obj_key)
            if existing is not None:
                return existing
            parent = (
                self.attach_room(room_id) if room_id else self.root
            )
            obj = ControlNode(
                key=obj_key,
                label=label or object_id,
                scope=OwnerScope.OBJECT,
                owner_id=object_id,
            )
            parent.add(obj)
            self._index_by_key[obj.key] = obj
            self._attach_owner_axes(obj, owner_id=object_id)
            return obj

    # ---- per-axis subtree lookup ----

    def controls_node(self, owner_id: str = "global") -> ControlNode:
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        return self._require(f"{owner_key}/controls")

    def actions_node(self, owner_id: str = "global") -> ControlNode:
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        return self._require(f"{owner_key}/actions")

    def shaders_node(self, owner_id: str = "global") -> ControlNode:
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        return self._require(f"{owner_key}/shaders")

    def inputs_node(self, owner_id: str = "global") -> ControlNode:
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        return self._require(f"{owner_key}/inputs")

    def outputs_node(self, owner_id: str = "global") -> ControlNode:
        owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
        return self._require(f"{owner_key}/outputs")

    def fifo_context(self, owner_id: str = "global") -> "FifoContext":
        """Build a :class:`FifoContext` view for the given owner.

        Endpoint ids are taken verbatim from each ``inputs/`` and
        ``outputs/`` child node's payload. Nodes whose payload does not
        carry a :class:`Fifo` are skipped (so legacy / placeholder
        endpoints registered without a FIFO instance are simply absent
        from the view).
        """
        with self._lock:
            inputs_axis = self.inputs_node(owner_id)
            outputs_axis = self.outputs_node(owner_id)
            inputs: dict[str, Fifo] = {}
            for node in inputs_axis.children:
                fifo = node.payload.get("fifo")
                if isinstance(fifo, Fifo):
                    inputs[str(node.payload.get("endpoint_id", node.key))] = fifo
            outputs: dict[str, Fifo] = {}
            for node in outputs_axis.children:
                fifo = node.payload.get("fifo")
                if isinstance(fifo, Fifo):
                    outputs[str(node.payload.get("endpoint_id", node.key))] = fifo
            return FifoContext(owner_id, inputs, outputs)

    # ---- attachment of leaf nodes ----

    def add_control_node(
        self,
        owner_id: str,
        node: ControlNode,
    ) -> ControlNode:
        parent = self.controls_node(owner_id)
        with self._lock:
            parent.add(node)
            self._index_by_key[node.key] = node
            if node.action_key:
                self._index_by_action[node.action_key] = node
        return node

    def add_action_node(
        self,
        owner_id: str,
        node: ControlNode,
    ) -> ControlNode:
        parent = self.actions_node(owner_id)
        with self._lock:
            parent.add(node)
            self._index_by_key[node.key] = node
            if node.action_key:
                self._index_by_action[node.action_key] = node
            if node.triangle_group:
                self._index_by_triangle_group[node.triangle_group] = node
        return node

    def add_shader_node(
        self,
        owner_id: str,
        node: ControlNode,
    ) -> ControlNode:
        parent = self.shaders_node(owner_id)
        with self._lock:
            parent.add(node)
            self._index_by_key[node.key] = node
            shader_id = str(node.payload.get("shader_id", node.key))
            self._index_by_shader[shader_id] = node
        return node

    def add_input_endpoint(
        self,
        owner_id: str,
        node: ControlNode,
    ) -> ControlNode:
        parent = self.inputs_node(owner_id)
        with self._lock:
            parent.add(node)
            self._index_by_key[node.key] = node
        return node

    def add_output_endpoint(
        self,
        owner_id: str,
        node: ControlNode,
    ) -> ControlNode:
        parent = self.outputs_node(owner_id)
        with self._lock:
            parent.add(node)
            self._index_by_key[node.key] = node
        return node

    # ---- lookup ----

    def find(self, key: str) -> Optional[ControlNode]:
        with self._lock:
            return self._index_by_key.get(str(key))

    def find_by_action_key(self, action_key: str) -> Optional[ControlNode]:
        with self._lock:
            return self._index_by_action.get(str(action_key))

    def find_by_triangle_group(self, group: str) -> Optional[ControlNode]:
        with self._lock:
            return self._index_by_triangle_group.get(str(group))

    def _require(self, key: str) -> ControlNode:
        node = self.find(key)
        if node is None:
            raise KeyError(f"control node not found: {key}")
        return node

    def iter_owner_ids(self) -> list[str]:
        """Return all owner ids that have FIFO endpoint axes attached.

        Yields ``"global"`` first (if it has any endpoints), then every
        ``object/<id>`` owner in insertion order.  Used by
        ``naive_graph.auto_wire_naive_network`` to discover unregistered
        FIFO endpoints without needing direct access to ``_index_by_key``.
        """
        with self._lock:
            result: list[str] = []
            for key, node in self._index_by_key.items():
                if node.scope is OwnerScope.OBJECT:
                    result.append(node.owner_id)
                elif node.scope is OwnerScope.GLOBAL and node.owner_id == "global":
                    result.insert(0, "global")
            # de-duplicate while preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for oid in result:
                if oid not in seen:
                    seen.add(oid)
                    deduped.append(oid)
            return deduped


_GRAPH: Optional[ControlGraph] = None


def get_control_graph() -> ControlGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = ControlGraph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

ActionCallable = Callable[..., Any]


@dataclass(slots=True)
class ActionBinding:
    action_key: str
    control_action: str = "invoke"
    hook: Optional[ActionCallable] = None
    script_path: str = ""
    hook_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, ActionBinding] = {}
        self._script_module_cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def ensure_action(
        self,
        action_key: str,
        *,
        control_action: str = "invoke",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ActionBinding:
        key = str(action_key)
        with self._lock:
            binding = self._bindings.get(key)
            if binding is None:
                binding = ActionBinding(
                    action_key=key,
                    control_action=str(control_action),
                    metadata=dict(metadata or {}),
                )
                self._bindings[key] = binding
                return binding
            if control_action:
                binding.control_action = str(control_action)
            if metadata:
                binding.metadata.update(metadata)
            return binding

    def register_callable(
        self,
        action_key: str,
        hook: ActionCallable,
        *,
        control_action: str = "invoke",
        metadata: Optional[dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> ActionBinding:
        if not callable(hook):
            raise TypeError("hook must be callable")
        with self._lock:
            binding = self.ensure_action(
                action_key,
                control_action=control_action,
                metadata=metadata,
            )
            if binding.hook is None or overwrite:
                binding.hook = hook
                binding.script_path = ""
                binding.hook_name = ""
            return binding

    def register_script(
        self,
        action_key: str,
        script_path: str,
        hook_name: str,
        *,
        control_action: str = "invoke",
        metadata: Optional[dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> ActionBinding:
        sp = str(script_path).strip()
        hn = str(hook_name).strip()
        if not sp:
            raise ValueError("script_path must not be empty")
        if not hn:
            raise ValueError("hook_name must not be empty")
        with self._lock:
            binding = self.ensure_action(
                action_key,
                control_action=control_action,
                metadata=metadata,
            )
            if binding.hook is None or overwrite:
                binding.hook = None
                binding.script_path = sp
                binding.hook_name = hn
            return binding

    def list_actions(self, prefix: str = "") -> list[str]:
        pfx = str(prefix)
        with self._lock:
            keys = sorted(self._bindings.keys())
        if not pfx:
            return keys
        return [k for k in keys if k.startswith(pfx)]

    def get_binding(self, action_key: str) -> Optional[ActionBinding]:
        with self._lock:
            return self._bindings.get(str(action_key))

    def dispatch(self, action_key: str, **context: Any) -> dict[str, Any]:
        """Synchronously invoke the hook bound to ``action_key``.

        The application normally goes through :func:`enqueue_action` so that
        hook execution happens on the worker thread; this synchronous form
        exists for tests, scripted setup, and inline programmatic use.
        """
        key = str(action_key)
        with self._lock:
            binding = self._bindings.get(key)
        if binding is None:
            return {"ok": False, "error": "unregistered_action", "action_key": key}

        hook = binding.hook or self._resolve_script_hook(binding)
        if hook is None:
            return {
                "ok": False,
                "error": "missing_hook",
                "action_key": key,
                "control_action": binding.control_action,
            }

        try:
            result = hook(
                action_key=key,
                control_action=binding.control_action,
                metadata=dict(binding.metadata),
                **context,
            )
            return {
                "ok": True,
                "action_key": key,
                "control_action": binding.control_action,
                "result": result,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": "hook_exception",
                "action_key": key,
                "control_action": binding.control_action,
                "exception": repr(exc),
            }

    def _resolve_script_hook(self, binding: ActionBinding) -> Optional[ActionCallable]:
        sp = str(binding.script_path or "").strip()
        hn = str(binding.hook_name or "").strip()
        if not sp or not hn:
            return None

        path = Path(sp)
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        cache_key = str(path)

        with self._lock:
            mod = self._script_module_cache.get(cache_key)
            if mod is None:
                spec = spec_from_file_location(f"action_script_{path.stem}", cache_key)
                if spec is None or spec.loader is None:
                    return None
                mod = module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._script_module_cache[cache_key] = mod

            hook = getattr(mod, hn, None)
            if callable(hook):
                binding.hook = hook
                return hook
            return None


_REGISTRY: Optional[ActionRegistry] = None


def get_action_registry() -> ActionRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ActionRegistry()
    return _REGISTRY


# ---------------------------------------------------------------------------
# Action dispatcher (queue + worker thread)
# ---------------------------------------------------------------------------

class _StopSentinel:
    pass


_STOP = _StopSentinel()


class ActionDispatcher:
    """Runs registered action hooks off the main loop.

    The main loop calls :meth:`enqueue` (cheap, lock-free queue put). A
    dedicated worker thread drains the queue and invokes the registered
    hook through the :class:`ActionRegistry`. This keeps event collection
    and OpenGL context ticking unblocked by potentially heavy action code.
    """

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        *,
        max_queue: int = 0,
    ) -> None:
        self._registry = registry or get_action_registry()
        self._queue: "queue.Queue[Any]" = queue.Queue(max_queue)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="ActionDispatcher",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 1.0) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._queue.put(_STOP)
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout)

    def enqueue(self, action_key: str, **context: Any) -> None:
        if not self._running:
            # Fail soft: dispatch synchronously so setup-time enqueues do
            # not get lost before the worker is started.
            self._registry.dispatch(action_key, **context)
            return
        self._queue.put((str(action_key), dict(context)))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if isinstance(item, _StopSentinel):
                return
            try:
                key, ctx = item
                self._registry.dispatch(key, **ctx)
            except Exception:
                # Worker must never die; swallow and continue.
                continue


_DISPATCHER: Optional[ActionDispatcher] = None


def get_dispatcher() -> ActionDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = ActionDispatcher()
    return _DISPATCHER


def enqueue_action(action_key: str, **context: Any) -> None:
    get_dispatcher().enqueue(action_key, **context)


def start_action_dispatcher() -> ActionDispatcher:
    disp = get_dispatcher()
    disp.start()
    return disp


def stop_action_dispatcher() -> None:
    if _DISPATCHER is not None:
        _DISPATCHER.stop()


# ---------------------------------------------------------------------------
# Conversion helpers (the only sanctioned producers of control nodes)
# ---------------------------------------------------------------------------

def slider_defs_to_knobs(
    defs: Iterable[tuple[Any, ...]],
    *,
    group: str,
    source_class: str,
) -> list[Any]:
    """Convert legacy 7-tuple slider definitions to KnobSpec instances."""
    out: list[Any] = []
    for row in defs:
        if len(row) < 7:
            continue
        key, label, lo, hi, default, is_log, live = row[:7]
        dtype = "float"
        if isinstance(default, int) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            dtype = "int"
        knob = KnobSpec(
            str(key),
            str(label),
            dtype,
            default,
            float(lo),
            float(hi),
            0.0,
            "",
            [],
            bool(is_log),
            group,
            ".4g",
            source_class,
            bool(not live),
            None,
        )
        out.append(knob)
    return out


def knobs_to_object_nodes(
    knobs: Iterable[Any],
    *,
    object_id: str,
    parent_key: str,
) -> list[ControlNode]:
    """Convert KnobSpec instances to station-control ``ControlNode`` peers.

    Each produced node is a leaf under the object's ``station_controls``
    peer subtree. An ``action_key`` of the form ``control/<parent>.<knob>``
    is registered into the singleton action registry so later dispatch can
    locate the binding without re-walking the graph.
    """
    registry = get_action_registry()
    graph = get_control_graph()
    # Ensure the object exists in the graph.
    graph.attach_object(object_id)

    nodes: list[ControlNode] = []
    for idx, knob in enumerate(knobs):
        knob_name = str(getattr(knob, "name", idx))
        key = f"{parent_key}.{knob_name}"
        label = str(getattr(knob, "label", knob_name))
        control_action = str(getattr(knob, "control_action", "set_knob"))
        action_key = f"control/{key}"

        registry.ensure_action(
            action_key,
            control_action=control_action,
            metadata={
                "origin": "station_control",
                "object_id": object_id,
                "parent_key": parent_key,
                "knob_name": knob_name,
                "source_class": str(getattr(knob, "source_class", "")),
            },
        )

        node = ControlNode(
            key=f"object/{object_id}/controls/{key}",
            label=label,
            scope=OwnerScope.CONTROLS,
            owner_id=object_id,
            knob=knob,
            action_key=action_key,
            control_action=control_action,
            payload={"parent_key": parent_key, "knob_name": knob_name},
        )
        graph.add_control_node(object_id, node)
        nodes.append(node)
    return nodes


def register_triangle_group_action(
    *,
    object_id: str,
    triangle_group: str,
    action_key: str,
    script_path: str = "",
    hook_name: str = "",
    hook: Optional[ActionCallable] = None,
    control_action: str = "invoke",
    metadata: Optional[dict[str, Any]] = None,
    label: str = "",
) -> ControlNode:
    """Attach a triangle-group action node under ``object_id`` and register
    its hook (callable or script path) in the action registry."""
    registry = get_action_registry()
    graph = get_control_graph()
    graph.attach_object(object_id)

    if hook is not None:
        registry.register_callable(
            action_key,
            hook,
            control_action=control_action,
            metadata=metadata,
        )
    elif script_path and hook_name:
        registry.register_script(
            action_key,
            script_path,
            hook_name,
            control_action=control_action,
            metadata=metadata,
        )
    else:
        registry.ensure_action(
            action_key,
            control_action=control_action,
            metadata=metadata,
        )

    node = ControlNode(
        key=f"object/{object_id}/actions/{triangle_group}",
        label=label or triangle_group,
        scope=OwnerScope.ACTIONS,
        owner_id=object_id,
        triangle_group=triangle_group,
        action_key=action_key,
        control_action=control_action,
        payload=dict(metadata or {}),
    )
    graph.add_action_node(object_id, node)
    return node


# ---------------------------------------------------------------------------
# FIFO endpoints (placeholders for the future networked compute subsystem)
# ---------------------------------------------------------------------------

class Fifo:
    """Thread-safe FIFO with its own RLock.

    Each input/output endpoint owns one ``Fifo``. Both the main-thread
    shader tick and the (future) compute-graph coordinator tick reach
    these FIFOs through a :class:`FifoContext`; every accessor takes the
    fifo's own lock, so the two ticks can safely run on different threads
    against disjoint endpoints, and any contention on a single endpoint
    is serialized at the finest possible granularity.

    Storage is opaque -- the FIFO carries whatever payload the producer
    writes (numpy array, OpenGL handle, dict of buffers, ...). Capacity
    of 0 means unbounded; otherwise the oldest item is dropped on push.
    """

    __slots__ = ("endpoint_id", "lock", "_q", "_capacity", "_dropped")

    def __init__(self, endpoint_id: str, *, capacity: int = 0) -> None:
        from collections import deque
        self.endpoint_id = str(endpoint_id)
        self.lock = threading.RLock()
        self._capacity = int(capacity)
        self._q = deque(maxlen=capacity if capacity > 0 else None)
        self._dropped = 0

    def push(self, item: Any) -> None:
        with self.lock:
            if self._capacity > 0 and len(self._q) == self._capacity:
                self._dropped += 1
            self._q.append(item)

    def pop(self) -> Any:
        with self.lock:
            return self._q.popleft() if self._q else None

    def peek(self) -> Any:
        with self.lock:
            return self._q[-1] if self._q else None

    def drain(self) -> list[Any]:
        with self.lock:
            items = list(self._q)
            self._q.clear()
            return items

    def __len__(self) -> int:
        with self.lock:
            return len(self._q)

    @property
    def dropped(self) -> int:
        with self.lock:
            return int(self._dropped)


class FifoContext:
    """Per-owner read/write view over input and output FIFOs.

    The naming convention is intentionally simple: every FIFO is
    referenced by its raw ``endpoint_id`` within its owner's scope. There
    is no global namespace mixing; ``inputs[name]`` and ``outputs[name]``
    are separate dicts so the same name can appear on both axes without
    ambiguity. Shader pre/run/post hooks and the compute-tick hook all
    receive the same ``FifoContext`` object for the owner they belong to,
    so referencing ``ctx.inputs['radar']`` from any of them resolves to
    the exact same lock-protected FIFO.
    """

    __slots__ = ("owner_id", "inputs", "outputs")

    def __init__(
        self,
        owner_id: str,
        inputs: dict[str, "Fifo"],
        outputs: dict[str, "Fifo"],
    ) -> None:
        self.owner_id = str(owner_id)
        self.inputs = inputs
        self.outputs = outputs

    def input(self, endpoint_id: str) -> "Fifo":
        return self.inputs[str(endpoint_id)]

    def output(self, endpoint_id: str) -> "Fifo":
        return self.outputs[str(endpoint_id)]

    def has_input(self, endpoint_id: str) -> bool:
        return str(endpoint_id) in self.inputs

    def has_output(self, endpoint_id: str) -> bool:
        return str(endpoint_id) in self.outputs


# ---------------------------------------------------------------------------
# Shader subsystem (cadence + flip buffers + bottom-up walk)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ShaderFlipBuffer:
    """A double-buffered output target a shader publishes into.

    The shader writes into ``write_index`` each tick; consumers read from
    the opposite slot. Buffer storage is opaque (texture id, FBO id, SSBO
    handle, CPU array, etc.); this struct only manages the rotation.
    """

    target_id: str
    slots: int = 2
    write_index: int = 0
    payloads: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slots < 1:
            self.slots = 1
        if not self.payloads:
            self.payloads = [None] * self.slots

    def write(self, payload: Any) -> int:
        idx = self.write_index % self.slots
        self.payloads[idx] = payload
        self.write_index = (self.write_index + 1) % self.slots
        return idx

    def latest(self) -> Any:
        # Most recently written slot (write_index points at next-to-write).
        return self.payloads[(self.write_index - 1) % self.slots]


@dataclass(slots=True)
class ShaderSpec:
    """Description of a custom shader registered in the graph.

    The spec carries everything the main-thread frame walker needs to
    decide whether to run the shader this tick and which global targets
    its outputs partially finalize.

    Hooks (all optional) all receive the same kwargs and have safe,
    lock-protected access to the owner's input/output FIFOs through the
    ``fifos`` :class:`FifoContext`:

    - ``pre_hook(spec, node, fifos, frame_index, dt) -> dict | None``
      Runs immediately before ``run`` on the same thread. May read or
      drain inputs and prepare buffers; whatever dict it returns is
      merged into the kwargs handed to ``run`` (so the pre-hook is the
      sanctioned channel for passing extra buffers down to the shader).
    - ``run(spec, node, fifos, frame_index, dt, **extra) -> dict | None``
      Invoked on the main thread (so it may touch the OpenGL context).
      Returns ``{target_id: payload}`` to be written into the matching
      flip buffers.
    - ``post_hook(spec, node, fifos, produced, frame_index, dt)``
      Runs immediately after ``run`` on the same thread. Typical use is
      pushing finished payloads onto output FIFOs.
    - ``compute_tick(spec, node, fifos, **kwargs)``
      Reserved exclusively for the future computational-graph
      coordinator. The shader walker NEVER calls this hook; it is fired
      only via :func:`dispatch_compute_tick` from a coordinator running
      on its own thread. FIFO access is still safe because each FIFO
      carries its own RLock.

    Cadence/targets:

    - ``min_period_s`` is the minimum wall-clock interval between
      ``run`` invocations (0.0 means "every frame"). It does not gate
      ``compute_tick`` -- the coordinator owns that cadence.
    - ``targets`` enumerates global buffer ids this shader claims
      partial responsibility for. After a successful ``run``, those ids
      are added to the frame's ``finalized_targets`` set so the final
      draw can gate fragments based on prior finalization.
    """

    shader_id: str
    targets: tuple[str, ...] = ()
    flip_slots: int = 2
    min_period_s: float = 0.0
    run: Optional[Callable[..., Any]] = None
    pre_hook: Optional[Callable[..., Any]] = None
    post_hook: Optional[Callable[..., Any]] = None
    compute_tick: Optional[Callable[..., Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def register_shader_node(
    *,
    owner_id: str,
    shader_id: str,
    run: Optional[Callable[..., Any]] = None,
    pre_hook: Optional[Callable[..., Any]] = None,
    post_hook: Optional[Callable[..., Any]] = None,
    compute_tick: Optional[Callable[..., Any]] = None,
    targets: Iterable[str] = (),
    min_period_s: float = 0.0,
    flip_slots: int = 2,
    metadata: Optional[dict[str, Any]] = None,
    label: str = "",
) -> ControlNode:
    """Attach a custom shader node under ``<owner>/shaders``.

    Owner is either ``"global"`` or an object id. The owner is created
    on demand; for objects this also creates the controls/actions/inputs/
    outputs peer subtrees. See :class:`ShaderSpec` for the meaning of the
    ``run`` / ``pre_hook`` / ``post_hook`` / ``compute_tick`` callbacks.
    """
    graph = get_control_graph()
    if owner_id != "global":
        graph.attach_object(owner_id)

    targets_t = tuple(str(t) for t in targets)
    spec = ShaderSpec(
        shader_id=str(shader_id),
        targets=targets_t,
        flip_slots=int(flip_slots),
        min_period_s=float(min_period_s),
        run=run,
        pre_hook=pre_hook,
        post_hook=post_hook,
        compute_tick=compute_tick,
        metadata=dict(metadata or {}),
    )

    flip_buffers: dict[str, ShaderFlipBuffer] = {
        tid: ShaderFlipBuffer(target_id=tid, slots=int(flip_slots))
        for tid in targets_t
    }

    owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
    node = ControlNode(
        key=f"{owner_key}/shaders/{shader_id}",
        label=label or shader_id,
        scope=OwnerScope.SHADERS,
        owner_id=owner_id,
        payload={
            "shader_id": str(shader_id),
            "spec": spec,
            "flip_buffers": flip_buffers,
            "last_run_time": 0.0,
            "last_run_frame": -1,
            "run_count": 0,
        },
    )
    graph.add_shader_node(owner_id, node)
    return node


def register_input_endpoint(
    *,
    owner_id: str,
    endpoint_id: str,
    capacity: int = 0,
    label: str = "",
    port_class: PortClass = PortClass.NAIVE_PRONG,
    port_role: PortRole = PortRole.STANDARD,
    metadata: Optional[dict[str, Any]] = None,
) -> ControlNode:
    """Reserve an input FIFO endpoint slot under ``<owner>/inputs``.

    ``port_class`` records the physical contact geometry; ``port_role``
    records the logical role the port plays inside its star.
    """
    graph = get_control_graph()
    if owner_id != "global":
        graph.attach_object(owner_id)
    owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
    fifo = Fifo(endpoint_id, capacity=int(capacity))
    pc = port_class if isinstance(port_class, PortClass) else PortClass(str(port_class))
    pr = port_role if isinstance(port_role, PortRole) else PortRole(str(port_role))
    node = ControlNode(
        key=f"{owner_key}/inputs/{endpoint_id}",
        label=label or endpoint_id,
        scope=OwnerScope.INPUTS,
        owner_id=owner_id,
        payload={
            "endpoint_id": str(endpoint_id),
            "fifo": fifo,
            "port_class": pc,
            "port_role": pr,
            **dict(metadata or {}),
        },
    )
    graph.add_input_endpoint(owner_id, node)
    return node


def register_output_endpoint(
    *,
    owner_id: str,
    endpoint_id: str,
    capacity: int = 0,
    label: str = "",
    port_class: PortClass = PortClass.NAIVE_PRONG,
    port_role: PortRole = PortRole.STANDARD,
    metadata: Optional[dict[str, Any]] = None,
) -> ControlNode:
    """Reserve an output FIFO endpoint slot under ``<owner>/outputs``.

    See :func:`register_input_endpoint`.
    """
    graph = get_control_graph()
    if owner_id != "global":
        graph.attach_object(owner_id)
    owner_key = "global" if owner_id == "global" else f"object/{owner_id}"
    fifo = Fifo(endpoint_id, capacity=int(capacity))
    pc = port_class if isinstance(port_class, PortClass) else PortClass(str(port_class))
    pr = port_role if isinstance(port_role, PortRole) else PortRole(str(port_role))
    node = ControlNode(
        key=f"{owner_key}/outputs/{endpoint_id}",
        label=label or endpoint_id,
        scope=OwnerScope.OUTPUTS,
        owner_id=owner_id,
        payload={
            "endpoint_id": str(endpoint_id),
            "fifo": fifo,
            "port_class": pc,
            "port_role": pr,
            **dict(metadata or {}),
        },
    )
    graph.add_output_endpoint(owner_id, node)
    return node


@dataclass(slots=True)
class ShaderFrameResult:
    """Per-frame summary handed to the final draw stage."""

    frame_index: int
    ran: list[str] = field(default_factory=list)        # shader_ids that ran
    skipped: list[str] = field(default_factory=list)    # shader_ids resting
    finalized_targets: set[str] = field(default_factory=set)
    flip_buffers: dict[str, ShaderFlipBuffer] = field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)


class ShaderFrameWalker:
    """Bottom-up shader walker for the main draw cycle.

    Each frame the main thread calls :meth:`tick`. The walker traverses
    every ``SHADERS`` subtree in post-order (children first, then the
    subtree root, then its owner's siblings, then its parent's level...).
    For every shader node it inspects the cadence (``min_period_s``); if
    the shader is rested, ``ShaderSpec.run`` is invoked, the returned
    payloads are written into the matching flip buffers, and each target
    id the shader claims is added to ``finalized_targets``. Regardless of
    whether the shader ran this tick, its already-published flip buffers
    are forwarded so the final draw can sample whatever was last left for
    each global buffer.

    The returned :class:`ShaderFrameResult` is the contract between this
    traversal and the final draw stage: the final draw uses
    ``finalized_targets`` to gate fragments, and ``flip_buffers`` to read
    the latest payload per target.
    """

    def __init__(self, graph: Optional[ControlGraph] = None) -> None:
        self._graph = graph or get_control_graph()
        self._monotonic: Callable[[], float] = __import__("time").monotonic

    def tick(self, *, frame_index: int, dt: float = 0.0) -> ShaderFrameResult:
        result = ShaderFrameResult(frame_index=int(frame_index))
        now = self._monotonic()

        for shader_node in self._iter_shaders_post_order():
            payload = shader_node.payload
            spec: ShaderSpec | None = payload.get("spec")
            if spec is None:
                continue

            # Always forward whatever flip buffers exist (even if resting).
            buffers: dict[str, ShaderFlipBuffer] = payload.get("flip_buffers", {})
            for tid, fb in buffers.items():
                # Last writer for a given target wins for this frame's view.
                result.flip_buffers[tid] = fb

            last_run = float(payload.get("last_run_time", 0.0))
            elapsed = now - last_run
            should_run = spec.run is not None and (
                spec.min_period_s <= 0.0 or elapsed >= spec.min_period_s
            )
            if not should_run:
                result.skipped.append(spec.shader_id)
                continue

            # Per-owner FIFO view shared by pre / run / post hooks.
            fifos = self._graph.fifo_context(shader_node.owner_id or "global")

            # ---- pre hook: may stage extra kwargs / buffers for run ----
            extra: dict[str, Any] = {}
            if spec.pre_hook is not None:
                try:
                    pre_out = spec.pre_hook(
                        spec=spec,
                        node=shader_node,
                        fifos=fifos,
                        frame_index=int(frame_index),
                        dt=float(dt),
                    )
                except Exception as exc:
                    result.errors.append((spec.shader_id, f"pre_hook: {exc!r}"))
                    continue
                if isinstance(pre_out, dict):
                    extra = pre_out

            # ---- run ----
            try:
                produced = spec.run(
                    spec=spec,
                    node=shader_node,
                    fifos=fifos,
                    frame_index=int(frame_index),
                    dt=float(dt),
                    **extra,
                )
            except Exception as exc:
                result.errors.append((spec.shader_id, repr(exc)))
                continue

            payload["last_run_time"] = now
            payload["last_run_frame"] = int(frame_index)
            payload["run_count"] = int(payload.get("run_count", 0)) + 1
            result.ran.append(spec.shader_id)

            if isinstance(produced, dict):
                for tid, value in produced.items():
                    fb = buffers.get(str(tid))
                    if fb is None:
                        fb = ShaderFlipBuffer(target_id=str(tid), slots=spec.flip_slots)
                        buffers[str(tid)] = fb
                    fb.write(value)
                    result.flip_buffers[str(tid)] = fb

            for tid in spec.targets:
                result.finalized_targets.add(str(tid))

            # ---- post hook: may publish onto output FIFOs ----
            if spec.post_hook is not None:
                try:
                    spec.post_hook(
                        spec=spec,
                        node=shader_node,
                        fifos=fifos,
                        produced=produced,
                        frame_index=int(frame_index),
                        dt=float(dt),
                    )
                except Exception as exc:
                    result.errors.append((spec.shader_id, f"post_hook: {exc!r}"))

        return result

    def _iter_shaders_post_order(self) -> Iterable[ControlNode]:
        """Yield every SHADERS leaf in bottom-up order.

        Owner traversal is post-order (deepest objects first, then their
        rooms, then global). Within each owner, the SHADERS subtree is
        yielded in post-order so child shader nodes are seen before any
        sibling parent shader nodes.
        """
        root = self._graph.root
        for owner in self._iter_owners_post_order(root):
            shaders_axis = self._find_axis_child(owner, OwnerScope.SHADERS)
            if shaders_axis is None:
                continue
            yield from self._iter_subtree_post_order(shaders_axis, skip_root=True)

    def _iter_owners_post_order(self, root: ControlNode) -> Iterable[ControlNode]:
        # Owners are: GLOBAL (the root itself) and OBJECT nodes. ROOM is
        # purely organizational and does not own subordinate axes.
        def walk(node: ControlNode) -> Iterable[ControlNode]:
            for child in node.children:
                if child.scope in (OwnerScope.ROOM, OwnerScope.OBJECT):
                    yield from walk(child)
            if node.scope in (OwnerScope.OBJECT, OwnerScope.GLOBAL):
                yield node
        yield from walk(root)

    def _find_axis_child(
        self,
        owner: ControlNode,
        scope: OwnerScope,
    ) -> Optional[ControlNode]:
        for child in owner.children:
            if child.scope is scope:
                return child
        return None

    def _iter_subtree_post_order(
        self,
        node: ControlNode,
        *,
        skip_root: bool = False,
    ) -> Iterable[ControlNode]:
        for child in node.children:
            yield from self._iter_subtree_post_order(child, skip_root=False)
        if not skip_root:
            yield node


_SHADER_WALKER: Optional[ShaderFrameWalker] = None


def get_shader_walker() -> ShaderFrameWalker:
    global _SHADER_WALKER
    if _SHADER_WALKER is None:
        _SHADER_WALKER = ShaderFrameWalker()
    return _SHADER_WALKER


def dispatch_compute_tick(shader_id: str, /, **kwargs: Any) -> Any:
    """Fire a shader node's ``compute_tick`` hook.

    This entry point exists exclusively for the future computational-graph
    coordinator. The main-thread shader walker NEVER calls it. The hook
    is invoked on whatever thread the caller is running on; the FIFO
    locks make the call safe against a concurrent shader tick on the
    same owner. Any extra ``kwargs`` (typically the coordinator's frame
    index, time budget, etc.) are forwarded verbatim.
    """
    graph = get_control_graph()
    node = graph.find_by_shader_id(shader_id) if hasattr(graph, "find_by_shader_id") else None
    if node is None:
        # Fall back: scan SHADERS axes for the id. Cheap; small N.
        for owner in graph.root.children:
            pass
        with graph._lock:  # type: ignore[attr-defined]
            node = graph._index_by_shader.get(str(shader_id))  # type: ignore[attr-defined]
    if node is None:
        raise KeyError(f"shader node not found: {shader_id}")
    spec: ShaderSpec | None = node.payload.get("spec")
    if spec is None or spec.compute_tick is None:
        return None
    fifos = graph.fifo_context(node.owner_id or "global")
    return spec.compute_tick(spec=spec, node=node, fifos=fifos, **kwargs)


__all__ = [
    "KnobSpec",
    "Panel",
    "OwnerScope",
    "ControlNode",
    "ControlGraph",
    "get_control_graph",
    "ActionBinding",
    "ActionRegistry",
    "get_action_registry",
    "ActionDispatcher",
    "get_dispatcher",
    "enqueue_action",
    "start_action_dispatcher",
    "stop_action_dispatcher",
    "slider_defs_to_knobs",
    "knobs_to_object_nodes",
    "register_triangle_group_action",
    "ShaderSpec",
    "ShaderFlipBuffer",
    "ShaderFrameResult",
    "ShaderFrameWalker",
    "get_shader_walker",
    "register_shader_node",
    "register_input_endpoint",
    "register_output_endpoint",
    "PortClass",
    "port_class_of",
    "PortRole",
    "port_role_of",
    "Fifo",
    "FifoContext",
    "dispatch_compute_tick",
    # iter_owner_ids is a method on ControlGraph; re-exported for convenience
]
