#!/usr/bin/env python3
"""Routing graph helpers and rack/port view construction."""
from __future__ import annotations

from analytic_shared import *  # noqa: F401,F403
import analytic_shared as _analytic_shared
import analytic_model as _analytic_model

def _import_all_from(module):
    globals().update({
        k: v for k, v in vars(module).items()
        if k != "_import_all_from" and not (k.startswith('__') and k.endswith('__'))
    })

_import_all_from(_analytic_model)
_import_all_from(_analytic_shared)

_PATCH_VIRTUAL_KEYS: tuple = (
    "__patch_tonic__",
    "__patch_seq__",
)
_ROUTER_UI_PREFIX: str = "__router__:"

_DEMO_VIRTUAL_LABELS: dict[str, str] = {
    "__patch_tonic__": "Tonic",
    "__patch_seq__": "Seq Pitch",
}

_DEMO_VIRTUAL_COLORS: dict[str, tuple[int, int, int]] = {
    "__patch_tonic__": (80, 180, 255),
    "__patch_seq__": (80, 220, 160),
}

def _control_connection_target_port_key(dst_key: str, param_path: str) -> str:
    return f"{dst_key}:param:{param_path}" if param_path else dst_key


def _control_connections_for_patch(patch: "AnalyticPatch") -> list[RackConnectionView]:
    g = patch.routing
    conns: list[RackConnectionView] = []
    for e in g.edges:
        if getattr(e, "edge_kind", "signal") == "signal":
            continue
        conns.append(RackConnectionView(
            src_port_key=e.src_port or e.src_key,
            dst_port_key=e.dst_port or e.dst_key,
            edge_kind=getattr(e, "edge_kind", "control"),
            remove_kind="control",
        ))
    for pe in g.param_edges:
        conns.append(RackConnectionView(
            src_port_key=pe.src_port or pe.src_key,
            dst_port_key=pe.dst_port or _control_connection_target_port_key(pe.dst_key, pe.param_path),
            edge_kind=getattr(pe, "edge_kind", "control"),
            remove_kind="param",
        ))
    for me in getattr(g, "meta_edges", []):
        conns.append(RackConnectionView(
            src_port_key=me.a_port or me.a_key,
            dst_port_key=me.b_port or me.b_key,
            edge_kind=getattr(me, "edge_kind", "meta"),
            remove_kind="meta",
        ))
    return conns


def _published_port_lookup(patch: "AnalyticPatch") -> dict[str, PublishedPort]:
    lookup: dict[str, PublishedPort] = {}
    for ports in _published_ports_for_patch(patch).values():
        for port in ports:
            lookup[port.key] = port
    return lookup


def _is_state_machine_owner(patch: "AnalyticPatch", owner_key: str) -> bool:
    mod = next((m for m in getattr(patch, "modules", []) if m.key == owner_key), None)
    return mod is not None and getattr(mod, "module_type", "") == "state_machine"


def _negotiated_lane_policy(src: PublishedPort, dst: PublishedPort) -> str:
    src_policy = str(getattr(src.tensor, "group_validity", "strict") or "strict")
    dst_policy = str(getattr(dst.tensor, "group_validity", "strict") or "strict")
    if src_policy == "remap" or dst_policy == "remap":
        return "remap"
    if src_policy == "reduce" or dst_policy == "reduce":
        return "reduce"
    if src_policy == "broadcast" or dst_policy == "broadcast":
        return "broadcast"
    return "strict"


def _negotiate_edge_transfer(
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> EdgeTransferSpec | None:
    if src_port.direction != "out" or dst_port.direction != "in":
        return None
    if src_port.tensor.analytic_only is not True or dst_port.tensor.analytic_only is not True:
        return None
    if str(src_port.tensor.dtype or "") != "complex128":
        return None
    if str(dst_port.tensor.dtype or "") != "complex128":
        return None

    if dst_port.domain != "param_target":
        if src_port.domain != dst_port.domain:
            if src_port.domain not in {"signal", "control"} or dst_port.domain not in {"signal", "control"}:
                return None

    src_group = str(src_port.tensor.parallel_group or "")
    dst_group = str(dst_port.tensor.parallel_group or "")
    lane_policy = _negotiated_lane_policy(src_port, dst_port)
    if src_group and dst_group and src_group != dst_group:
        if not (src_port.negotiates_group_validity or dst_port.negotiates_group_validity):
            return None
        if lane_policy == "strict":
            lane_policy = "remap"

    src_lanes = max(1, int(src_port.tensor.lane_count or 1))
    dst_lanes = max(1, int(dst_port.tensor.lane_count or 1))
    src_dynamic = int(src_port.tensor.lane_count or 0) == 0
    dst_dynamic = int(dst_port.tensor.lane_count or 0) == 0

    if src_port.tensor.batch_axes != dst_port.tensor.batch_axes:
        if not (src_port.tensor.batchable and dst_port.tensor.batchable):
            return None
    if src_port.tensor.batch_axes == dst_port.tensor.batch_axes:
        batch_policy = "strict"
    elif src_port.tensor.batch_axes < dst_port.tensor.batch_axes:
        batch_policy = "broadcast"
    else:
        batch_policy = "reduce"

    reduction = ""
    if dst_port.domain == "param_target":
        transfer_policy = "remap" if lane_policy == "remap" else "identity"
    elif src_dynamic or dst_dynamic:
        transfer_policy = "broadcast"
        if lane_policy == "strict":
            lane_policy = "broadcast"
    elif src_lanes == dst_lanes:
        transfer_policy = "identity"
    elif src_lanes == 1 and dst_lanes > 1:
        transfer_policy = "broadcast"
        if lane_policy == "strict":
            lane_policy = "broadcast"
    elif src_lanes > 1 and dst_lanes == 1:
        transfer_policy = "reduce"
        if lane_policy == "strict":
            lane_policy = "reduce"
        reduction = "mean"
    else:
        transfer_policy = "remap"
        lane_policy = "remap"

    return EdgeTransferSpec(
        transfer_policy=transfer_policy,
        cable_count=max(src_lanes, dst_lanes, 1),
        batch_policy=batch_policy,
        lane_policy=lane_policy,
        reduction=reduction,
        analytic_only=True,
    )


def _negotiate_metaedge_transfer(
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> tuple[EdgeTransferSpec, EdgeTransferSpec] | None:
    fwd = _negotiate_edge_transfer(src_port, dst_port)
    if fwd is None:
        return None

    rev_policy = "identity"
    rev_lane = fwd.lane_policy
    rev_batch = fwd.batch_policy
    if fwd.transfer_policy == "broadcast":
        rev_policy = "reduce"
        if rev_lane == "strict":
            rev_lane = "reduce"
    elif fwd.transfer_policy == "reduce":
        rev_policy = "broadcast"
        if rev_lane == "strict":
            rev_lane = "broadcast"
    elif fwd.transfer_policy == "remap":
        rev_policy = "remap"

    rev = EdgeTransferSpec(
        transfer_policy=rev_policy,
        cable_count=fwd.cable_count,
        batch_policy=rev_batch,
        lane_policy=rev_lane,
        reduction="mean" if rev_policy == "reduce" else "",
        analytic_only=True,
    )
    return fwd, rev


def _ports_compatible(src: PublishedPort, dst: PublishedPort) -> bool:
    return _negotiate_edge_transfer(src, dst) is not None


def _control_remove_connections_for_port(patch: "AnalyticPatch", port_key: str) -> None:
    patch.routing.edges = [
        e for e in patch.routing.edges
        if (e.src_port or e.src_key) != port_key
        and (e.dst_port or e.dst_key) != port_key
    ]
    patch.routing.param_edges = [
        pe for pe in patch.routing.param_edges
        if (pe.src_port or pe.src_key) != port_key
        and (pe.dst_port or _control_connection_target_port_key(pe.dst_key, pe.param_path)) != port_key
    ]
    patch.routing.meta_edges = [
        me for me in getattr(patch.routing, "meta_edges", [])
        if (me.a_port or me.a_key) != port_key
        and (me.b_port or me.b_key) != port_key
    ]


def _control_add_connection(
    patch: "AnalyticPatch",
    src_port: PublishedPort,
    dst_port: PublishedPort,
) -> bool:
    transfer_spec = _negotiate_edge_transfer(src_port, dst_port)
    if transfer_spec is None:
        return False

    if (
        dst_port.domain != "param_target"
        and _is_state_machine_owner(patch, src_port.owner_key)
        and _is_state_machine_owner(patch, dst_port.owner_key)
    ):
        meta_specs = _negotiate_metaedge_transfer(src_port, dst_port)
        if meta_specs is None:
            return False
        a_to_b, b_to_a = meta_specs
        patch.routing.meta_edges.append(MetaEdge(
            a_key=src_port.owner_key,
            b_key=dst_port.owner_key,
            a_port=src_port.key,
            b_port=dst_port.key,
            edge_kind="meta",
            semantic_role=src_port.semantic_role or dst_port.semantic_role or src_port.tensor.semantic_role or dst_port.tensor.semantic_role,
            channel_count=max(1, int(src_port.tensor.lane_count or dst_port.tensor.lane_count or 1)),
            tensor_contract=src_port.tensor.to_contract(),
            a_to_b_transfer=a_to_b,
            b_to_a_transfer=b_to_a,
        ))
        return True

    if dst_port.domain == "param_target":
        patch.routing.param_edges.append(ParamEdge(
            src_key=src_port.key,
            dst_key=dst_port.owner_key,
            weight=1.0,
            extractor="magnitude",
            delay_samples=0,
            param_path=dst_port.param_path,
            src_port=src_port.key,
            dst_port=dst_port.key,
            edge_kind="control",
            projection_policy=str(getattr(dst_port, "projection_policy", "") or "magnitude_mean"),
            tensor_contract=src_port.tensor.to_contract(),
            transfer_spec=transfer_spec,
        ))
        return True

    patch.routing.add_node(src_port.key)
    patch.routing.add_node(dst_port.key)
    patch.routing.edges.append(RoutingEdge(
        src_key=src_port.key,
        dst_key=dst_port.key,
        weight=1.0,
        angle_rad=0.0,
        delay_s=0.0,
        edge_kind="control",
        src_port=src_port.key,
        dst_port=dst_port.key,
        tensor_contract=src_port.tensor.to_contract(),
        transfer_spec=transfer_spec,
    ))
    return True


def _make_param_target_port(
    owner_key: str,
    owner_label: str,
    param_path: str,
    *,
    group: str = "Params",
    color: tuple[int, int, int] = (180, 140, 220),
    tensor_rank: int = 0,
    lane_count: int = 1,
    parallel_group: str = "",
    semantic_role: str = "param_target",
    projection_policy: str = "magnitude_mean",
) -> PublishedPort:
    return PublishedPort(
        key=f"{owner_key}:param:{param_path}",
        label=f"{owner_label}.{param_path}",
        direction="in",
        domain="param_target",
        owner_key=owner_key,
        group=group,
        param_path=param_path,
        color=color,
        tensor=PortTensorSpec(
            tensor_rank=tensor_rank,
            lane_count=lane_count,
            group_validity="strict",
            parallel_group=parallel_group,
            semantic_role=semantic_role,
            batchable=True,
        ),
        semantic_role=semantic_role,
        projection_policy=projection_policy,
        negotiates_group_validity=True,
    )


def _published_ports_for_voice(v: "AnalyticVoice") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=v.key,
            label=v.label,
            direction="out",
            domain="signal",
            owner_key=v.key,
            group="Signal",
            color=tuple(v.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="voice_signal"),
            negotiates_group_validity=True,
        ),
    ]
    for path in (
        "amplitude",
        "phase_origin",
        "chirp.f_delta_start",
        "chirp.f_delta_end",
        "chirp.tau",
        "chirp.chirp_power",
        "harmonic_brightness",
        "harmonic_warp_strength",
    ):
        ports.append(_make_param_target_port(
            v.key, v.label, path,
            color=tuple(v.color[:3]),
            parallel_group="voice_param_batch",
        ))
    return ports


def _published_ports_for_mixer(m: "AnalyticMixer") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=m.key,
            label=m.label,
            direction="out",
            domain="signal",
            owner_key=m.key,
            group="Signal",
            color=tuple(m.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="mixer_signal"),
            negotiates_group_validity=True,
        ),
    ]
    for path in ("projection_active",):
        ports.append(_make_param_target_port(
            m.key, m.label, path,
            group="Mixer",
            color=tuple(m.color[:3]),
            parallel_group="mixer_param_batch",
        ))
    return ports


def _published_ports_for_param_node(pn: "ParamNode") -> list[PublishedPort]:
    ports: list[PublishedPort] = [
        PublishedPort(
            key=pn.key,
            label=pn.label,
            direction="in",
            domain="control",
            owner_key=pn.key,
            group="Param Node",
            color=tuple(pn.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="param_control_in"),
            negotiates_group_validity=True,
        ),
        PublishedPort(
            key=f"{pn.key}:out",
            label=f"{pn.label}.out",
            direction="out",
            domain="control",
            owner_key=pn.key,
            group="Param Node",
            color=tuple(pn.color[:3]),
            tensor=PortTensorSpec(tensor_rank=1, lane_count=0, parallel_group="param_control_out"),
            negotiates_group_validity=True,
        ),
    ]
    for tgt in getattr(pn, "targets", []):
        path = str(tgt.get("attr", "") or "")
        if path:
            ports.append(_make_param_target_port(
                pn.key,
                pn.label,
                path,
                group="Targets",
                color=tuple(pn.color[:3]),
                parallel_group="param_target_batch",
            ))
    return ports


def _published_ports_for_module(mod: "AnalyticModule") -> list[PublishedPort]:
    ports: list[PublishedPort] = []
    base_color = tuple(mod.color[:3])
    ports.append(PublishedPort(
        key=mod.key,
        label=mod.label,
        direction="out",
        domain="signal",
        owner_key=mod.key,
        group="Signal",
        color=base_color,
        tensor=PortTensorSpec(
            tensor_rank=1,
            lane_count=0,
            parallel_group=f"module_signal:{mod.module_type}",
            semantic_role="signal",
        ),
        semantic_role="signal",
        negotiates_group_validity=True,
    ))
    if mod.module_type == "state_machine":
        for bundle in getattr(mod, "sm_bundle_ports", []):
            bundle_name = str(bundle.get("name", "") or "").strip()
            if not bundle_name:
                continue
            direction = str(bundle.get("direction", "out") or "out")
            domain = str(bundle.get("domain", "control") or "control")
            semantic_role = str(bundle.get("semantic_role", bundle_name) or bundle_name)
            channel_dims = [int(x) for x in bundle.get("channel_dims", [])]
            ports.append(PublishedPort(
                key=f"{mod.key}:{bundle_name}",
                label=f"{mod.label}.{bundle_name}",
                direction=direction,
                domain=domain,
                owner_key=mod.key,
                group=str(bundle.get("group", "SM Bundle") or "SM Bundle"),
                color=base_color,
                tensor=PortTensorSpec(
                    tensor_rank=max(0, int(bundle.get("tensor_rank", max(1, len(channel_dims))))),
                    lane_count=max(0, int(bundle.get("lane_count", 0))),
                    batch_axes=max(0, int(bundle.get("batch_axes", 1))),
                    parallel_group=str(bundle.get("parallel_group", f"sm_bundle:{mod.key}:{bundle_name}") or f"sm_bundle:{mod.key}:{bundle_name}"),
                    group_validity=str(bundle.get("group_validity", "remap") or "remap"),
                    semantic_role=semantic_role,
                    channel_dims=channel_dims,
                ),
                semantic_role=semantic_role,
                negotiates_group_validity=True,
            ))
        # Primary SM signal outputs
        for out_key in mod.sm_out_keys():
            ports.append(PublishedPort(
                key=out_key,
                label=out_key,
                direction="out",
                domain="signal",
                owner_key=mod.key,
                group="SM Signal",
                color=base_color,
                tensor=PortTensorSpec(
                    tensor_rank=1,
                    lane_count=max(0, int(getattr(mod, "sm_n_items", 1))),
                    parallel_group=f"sm_signal:{mod.key}",
                    semantic_role="sm_signal",
                ),
                semantic_role="sm_signal",
                negotiates_group_validity=True,
            ))
        # Declared control feedback ports — one per item/var pair to keep them
        # unique and compatible with the unified graph.
        for item in getattr(mod, "sm_items", []):
            for var in getattr(mod, "sm_vars", []):
                ctrl_key = f"{mod.key}.ctrl.{item}.{var}"
                ports.append(PublishedPort(
                    key=ctrl_key,
                    label=ctrl_key,
                    direction="out",
                    domain="control",
                    owner_key=mod.key,
                    group="SM Control",
                    color=base_color,
                    tensor=PortTensorSpec(
                        tensor_rank=1,
                        lane_count=max(0, int(getattr(mod, "sm_n_items", 1))),
                        parallel_group=f"sm_ctrl:{mod.key}",
                        semantic_role="sm_control",
                    ),
                    semantic_role="sm_control",
                    negotiates_group_validity=True,
                ))
    for path in ("rate_hz", "depth", "phase_offset", "sm_n_items"):
        if hasattr(mod, path.split(".")[0]):
            ports.append(_make_param_target_port(
                mod.key,
                mod.label,
                path,
                group="Module",
                color=base_color,
                parallel_group=f"module_param:{mod.module_type}",
            ))
    return ports


def _published_ports_for_router(router: "RouterInstance") -> list[PublishedPort]:
    label = f"{router.label} [{router.router_type}]"
    base_color = {
        "voice_router": (90, 160, 230),
        "instrument": (110, 200, 150),
        "master": (230, 180, 90),
    }.get(router.router_type, (160, 160, 180))
    return [
        _make_param_target_port(
            router.key, label, "feedback.enabled",
            group="Router", color=base_color, parallel_group="router_param_batch"),
        _make_param_target_port(
            router.key, label, "feedback.decay",
            group="Router", color=base_color, parallel_group="router_param_batch"),
        _make_param_target_port(
            router.key, label, "feedback.max_iterations",
            group="Router", color=base_color, parallel_group="router_param_batch"),
    ]


def _published_ports_for_patch_virtuals(patch: "AnalyticPatch") -> list[PublishedPort]:
    ports: list[PublishedPort] = []
    for key in _PATCH_VIRTUAL_KEYS:
        color = _DEMO_VIRTUAL_COLORS.get(key, (120, 160, 180))
        ports.append(PublishedPort(
            key=key,
            label=_routing_node_label(key, patch),
            direction="out",
            domain="pitch",
            owner_key="__patch_context__",
            group="Patch Context",
            color=color,
            tensor=PortTensorSpec(
                tensor_rank=1,
                lane_count=0,
                parallel_group="patch_context_stream",
                semantic_role="patch_context",
            ),
            semantic_role="patch_context",
            negotiates_group_validity=True,
        ))
    return ports


def _published_ports_for_patch(patch: "AnalyticPatch") -> dict[str, list[PublishedPort]]:
    out: dict[str, list[PublishedPort]] = {}
    out["__patch_context__"] = _published_ports_for_patch_virtuals(patch)
    for v in patch.voices:
        out[v.key] = _published_ports_for_voice(v)
    for m in patch.mixers:
        out[m.key] = _published_ports_for_mixer(m)
    for mod in patch.modules:
        out[mod.key] = _published_ports_for_module(mod)
    for cs in patch.controls:
        ports: list[PublishedPort] = []
        for sl in cs.sliders:
            ports.append(PublishedPort(
                key=sl.key,
                label=f"{cs.label}/{sl.label}",
                direction="out",
                domain="control",
                owner_key=cs.key,
                group="Controls",
                color=tuple(cs.color[:3]),
                tensor=PortTensorSpec(tensor_rank=0, lane_count=1, parallel_group="control_scalar"),
            ))
        out[cs.key] = ports
    for pn in patch.param_nodes:
        out[pn.key] = _published_ports_for_param_node(pn)
    for router in getattr(patch, "routers", []):
        out[_router_ui_key(router.key)] = _published_ports_for_router(router)
    return out


def _rack_device_views_for_patch(patch: "AnalyticPatch") -> list[RackDeviceView]:
    published = _published_ports_for_patch(patch)
    device_order: list[tuple[str, str, tuple[int, int, int], str]] = []
    device_order.append(("__patch_context__", "Patch Context", (90, 170, 190), "patch"))
    for v in patch.voices:
        device_order.append((v.key, v.label, tuple(v.color[:3]), "voice"))
    for mod in patch.modules:
        device_order.append((mod.key, mod.label, tuple(mod.color[:3]), f"module:{mod.module_type}"))
    for cs in patch.controls:
        device_order.append((cs.key, cs.label, tuple(cs.color[:3]), "control"))
    for pn in patch.param_nodes:
        device_order.append((pn.key, pn.label, tuple(pn.color[:3]), "param"))
    for mix in patch.mixers:
        device_order.append((mix.key, mix.label, tuple(mix.color[:3]), "mixer"))
    for router in getattr(patch, "routers", []):
        device_order.append((_router_ui_key(router.key), router.label,
                             {
                                 "voice_router": (90, 160, 230),
                                 "instrument": (110, 200, 150),
                                 "master": (230, 180, 90),
                             }.get(router.router_type, (160, 160, 180)),
                             f"router:{router.router_type}"))

    rack: list[RackDeviceView] = []
    x_slot = 0
    y_u = 0
    max_cols = 6
    for device_key, label, color, kind in device_order:
        ports = list(published.get(device_key, []))
        width = max(1, min(4, (len(ports) + 7) // 8))
        height = max(1, min(4, (len(ports) + width * 7) // max(width * 8, 1)))
        if x_slot + width > max_cols:
            x_slot = 0
            y_u += 4
        port_views: list[RackPortView] = []
        cols = max(1, width * 4)
        for pi, port in enumerate(ports):
            port_views.append(RackPortView(
                port=port,
                local_x=8 + (pi % cols) * 8,
                local_y=12 + (pi // cols) * 10,
                radius=3,
            ))
        rack.append(RackDeviceView(
            device_key=device_key,
            label=label,
            device_kind=kind,
            color=color,
            rack_u=height,
            rack_w=width,
            grid_x=x_slot,
            grid_y=y_u,
            ports=port_views,
        ))
        x_slot += width
    return rack
def _routing_edge_attr_specs(patch: "AnalyticPatch") -> list[tuple[str, str]]:
    """Return per-edge pseudo-attrs for routing-grid knobs with readable labels."""
    specs: list[tuple[str, str]] = []
    node_keys = _patch_node_keys(patch)
    for src_key in node_keys:
        src_lbl = _routing_node_label(src_key, patch)
        for dst_key in node_keys:
            dst_lbl = _routing_node_label(dst_key, patch)
            specs.append((f"routing.mix::{src_key}::{dst_key}",
                          f"Mix / {src_lbl} -> {dst_lbl}"))
            specs.append((f"routing.angle::{src_key}::{dst_key}",
                          f"Angle / {src_lbl} -> {dst_lbl}"))
            specs.append((f"routing.delay::{src_key}::{dst_key}",
                          f"Delay / {src_lbl} -> {dst_lbl}"))
    return specs


def _parse_routing_edge_attr(attr: str) -> "tuple[str, str, str] | None":
    prefix, sep, rest = attr.partition("::")
    if not sep or prefix not in {"routing.mix", "routing.angle", "routing.delay"}:
        return None
    src_key, sep2, dst_key = rest.partition("::")
    if not sep2 or not src_key or not dst_key:
        return None
    kind = prefix.split(".", 1)[1]
    return kind, src_key, dst_key


def _param_target_node_specs(patch: "AnalyticPatch") -> list[tuple[str, str]]:
    """Return selectable ParamNode target nodes with readable labels."""
    specs: list[tuple[str, str]] = []
    specs.extend((v.key, v.label) for v in patch.voices)
    specs.extend((l.key, f"~ {l.label}") for l in patch.lfos)
    for mod in patch.modules:
        specs.append((mod.key, f"\u2B21 {mod.label}"))
        if mod.module_type == "interaural":
            specs.append((mod.ch1_key(), f"\u2B21 {mod.label} ch1"))
            specs.append((mod.ch2_key(), f"\u2B21 {mod.label} ch2"))
    for mix in patch.mixers:
        specs.append((mix.key, f"\u2261 {mix.label}"))
    return specs


def _param_target_attr_specs(patch: "AnalyticPatch", node_key: str) -> list[tuple[str, str]]:
    """Return (raw_attr, display_label) choices for a ParamNode target node."""
    if not node_key:
        return []
    voice = next((v for v in patch.voices if v.key == node_key), None)
    if voice is not None:
        voice_labels = {
            "freq_hz": "Frequency",
            "amplitude": "Amplitude",
            "semitone_offset": "Semitone Offset",
            "chirp.f_delta_start": "Chirp Start",
            "chirp.f_delta_end": "Chirp End",
            "adsr.attack": "Attack",
            "adsr.decay": "Decay",
            "adsr.sustain": "Sustain",
            "adsr.release": "Release",
            "fm.depth_hz": "FM Depth Hz",
            "fm.depth_amp": "FM Depth Amp",
            "am.depth_hz": "AM Depth Hz",
            "am.depth_amp": "AM Depth Amp",
            "harmonic_brightness": "Harmonic Brightness",
            "harmonic_warp_strength": "Harmonic Warp",
            "harmonic_count": "Harmonic Count",
            "granular.grain_density_hz": "Grain Density",
            "granular.grain_duration_s": "Grain Duration",
            "granular.grain_scatter": "Grain Scatter",
            "granular.grain_pitch_scatter": "Grain Pitch Scatter",
            "granular.grain_manifold_mix": "Grain Manifold Mix",
            "granular.grain_amplitude_jitter": "Grain Amp Jitter",
        }
        return [(attr, voice_labels.get(attr, attr)) for attr in _VOICE_PARAM_ATTRS]
    lfo = next((l for l in patch.lfos if l.key == node_key), None)
    if lfo is not None:
        labels = _knob_label_map(LFODefinition.knobs())
        return [(k.name, labels.get(k.name, k.name)) for k in LFODefinition.knobs()
                if k.dtype in ("float", "int", "bool", "choice")]
    mod = next((m for m in patch.modules
                if m.key == node_key or
                (m.module_type == "interaural" and node_key in (m.ch1_key(), m.ch2_key()))), None)
    if mod is not None:
        labels = _knob_label_map(AnalyticModule.knobs())
        attrs = _module_param_attrs(mod.module_type)
        return [(attr, labels.get(attr, attr)) for attr in attrs]
    mix = next((m for m in patch.mixers if m.key == node_key), None)
    if mix is not None:
        return _routing_edge_attr_specs(patch)
    return []
def _system_output_keys(patch: "AnalyticPatch") -> list[str]:
    return patch.system_audio.output_keys() if getattr(patch, "system_audio", None) else []


def _system_input_keys(patch: "AnalyticPatch") -> list[str]:
    return patch.system_audio.input_keys() if getattr(patch, "system_audio", None) else []


def _router_ui_key(router_key: str) -> str:
    return f"{_ROUTER_UI_PREFIX}{router_key}"


def _router_key_from_ui_key(ui_key: str) -> str:
    return str(ui_key)[len(_ROUTER_UI_PREFIX):] if str(ui_key).startswith(_ROUTER_UI_PREFIX) else ""


def _router_instance_for_active_key(
    patch: "AnalyticPatch",
    active_key: str,
) -> "RouterInstance | None":
    router_key = _router_key_from_ui_key(active_key)
    if not router_key:
        return None
    return next((r for r in getattr(patch, "routers", []) if r.key == router_key), None)


def _active_routing_graph_and_instance(
    patch: "AnalyticPatch",
    active_key: str = "",
) -> "tuple[RoutingGraph, RouterInstance | None]":
    router = _router_instance_for_active_key(patch, active_key)
    if router is not None:
        return router.graph, router
    return patch.routing, None


def _filtered_router_edges(
    graph: RoutingGraph,
    *,
    router_key: str = "",
    router_type: str = "",
) -> list[RoutingEdge]:
    edges = list(graph.edges)
    if router_key:
        edges = [e for e in edges if e.router_key in {"", router_key}]
    if router_type:
        edges = [
            e for e in edges
            if graph.is_source_in_router(e.src_key, router_type)
            and graph.is_sink_in_router(e.dst_key, router_type)
        ]
    return edges


def _routing_grid_node_keys(
    patch: "AnalyticPatch",
    active_key: str = "",
) -> list[str]:
    graph, router = _active_routing_graph_and_instance(patch, active_key)
    if router is None:
        return _patch_node_keys(patch)

    edges = _filtered_router_edges(
        graph,
        router_key=router.key,
        router_type=router.router_type,
    )
    referenced: set[str] = set()
    for e in edges:
        referenced.add(e.src_key)
        referenced.add(e.dst_key)

    canonical = _patch_node_keys(patch)
    router_type = str(getattr(router, "router_type", "") or "")
    graph_nodes = set(graph.node_keys())
    result: list[str] = []
    for key in canonical:
        if key not in graph_nodes and key not in referenced:
            continue
        if not router_type:
            if key in referenced or key in graph_nodes:
                result.append(key)
            continue
        if (
            key in referenced
            or graph.is_source_in_router(key, router_type)
            or graph.is_sink_in_router(key, router_type)
        ):
            result.append(key)
    extras = [
        key for key in graph.node_keys()
        if key not in result and (
            key in referenced
            or not router_type
            or graph.is_source_in_router(key, router_type)
            or graph.is_sink_in_router(key, router_type)
        )
    ]
    result.extend(extras)
    return result


def _runtime_performer_keys(patch: "AnalyticPatch") -> list[str]:
    keys: list[str] = []
    for pt in getattr(patch, "parts", []):
        for ch in getattr(pt, "chairs", []):
            for pf in getattr(ch, "performers", []):
                if getattr(pf, "key", ""):
                    keys.append(pf.key)
    return keys


def _sanitize_system_io_edges(g: RoutingGraph, patch: "AnalyticPatch") -> None:
    sys_in = set(_system_input_keys(patch))
    sys_out = set(_system_output_keys(patch))
    if not sys_in and not sys_out:
        return
    g.edges = [
        e for e in g.edges
        if e.src_key not in sys_out and e.dst_key not in sys_in
    ]

def _patch_node_keys(
    patch: "AnalyticPatch",
    include_params:   bool = True,
    include_controls: bool = True,
    include_performers: bool = False,
) -> list:
    """Canonical ordered node key list:
    virtual-patch-nodes -> voices -> LFOs -> module signal nodes -> controls -> mixers -> param_nodes.

    The two virtual patch nodes (__patch_tonic__, __patch_seq__) carry Hz
    pitch-domain signals (not audio) and are excluded from auto-mix routing.
    """
    keys = list(_PATCH_VIRTUAL_KEYS)
    keys += _system_input_keys(patch)
    if include_performers:
        keys += _runtime_performer_keys(patch)
    keys += [v.key for v in patch.voices]
    keys += [l.key for l in patch.lfos]
    for m in patch.modules:
        if m.module_type == "interaural":
            keys.append(m.ch1_key())
            keys.append(m.ch2_key())
        elif m.module_type == "lfo" and m.lfo_channels:
            keys.append(m.key)
            for i in range(len(m.lfo_channels)):
                keys.append(m.lfo_ch_key(i))
        elif m.module_type == "state_machine":
            keys.append(m.key)
            for k in m.sm_out_keys():
                keys.append(k)
        else:
            keys.append(m.key)
    if include_controls:
        for cs in patch.controls:
            keys += [sl.key for sl in cs.sliders]
    keys += [m.key for m in patch.mixers]
    keys += _system_output_keys(patch)
    if include_params:
        keys += [pn.key for pn in patch.param_nodes]
    return keys


# ---------------------------------------------------------------------------
# EditorCanvas — center view with PlotWidget + GL interactive overlay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# RoutingGridView — center-view knob matrix for signal routing
# ---------------------------------------------------------------------------

_SNAP_12TH = 1.0 / 12.0   # semitone-based snap increment for routing knobs

# Popular tuning roots used as snap points on the tuning.root_hz slider.
# Covers: A2/A3/A4 series (110 / 220 / 432 / 440 / 466.16 Hz),
#         baroque A415, French baroque A392,
#         C4 in standard tuning (261.626) and philosophical C (256).
_ROOT_HZ_SNAPS: tuple[float, ...] = (
    110.0,       # A2  — standard
    130.813,     # C3  — standard (A=440)
    220.0,       # A3  — standard
    256.0,       # C4  — philosophical / Verdi
    261.626,     # C4  — standard (A=440)
    392.0,       # A392 — French baroque
    415.0,       # A415 — baroque
    432.0,       # A432 — alternative
    440.0,       # A440 — ISO 16 standard concert pitch
    466.16,      # A466 — Chorton (German high baroque)
)
_ROOT_HZ_SNAP_TOL = 0.015   # ±1.5 % relative tolerance

# Standard audio sample rates — used for slider snapping / tick marks
_COMMON_SAMPLE_RATES: tuple[int, ...] = (
    8000, 11025, 16000, 22050, 32000,
    44100, 48000, 88200, 96000, 176400, 192000,
)

# Western chromatic pitch-class names (12-TET / MIDI convention)
_NOTE_NAMES_12 = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# 22-śruti Sargam names (Bharata / Natya Shastra ordering).
# Index = śruti number 0-21 (chromatic, from Sa).
_SRUTI_NAMES_22 = (
    "Sa",       #  0 tonic
    "koRe",     #  1 komal Re (ek śruti)
    "koRe₂",    #  2 komal Re (do śruti)
    "Re₁",      #  3 Ri-1 (tri śruti)
    "Re",       #  4 shuddha Re / chatur-śruti Ri
    "koGa₁",    #  5 sadharana Ga low
    "koGa",     #  6 sadharana Ga / komal Ga
    "Ga",       #  7 antara Ga
    "Ga₂",      #  8 shuddha Ga-2
    "Ma",       #  9 shuddha Ma (perfect 4th)
    "Ma₂",      # 10
    "tiMa₁",    # 11 tivra Ma-1
    "tiMa",     # 12 tivra Ma-2 (tritone)
    "Pa",       # 13 perfect 5th
    "koDha",    # 14 komal Dha (ek śruti)
    "koDha₂",   # 15 komal Dha (do śruti)
    "Dha₁",     # 16 Dha-1
    "Dha",      # 17 shuddha Dha / chatur-śruti
    "koNi₁",    # 18 komal Ni (ek śruti)
    "koNi",     # 19 kaisiki Ni
    "Ni₁",      # 20 kakali Ni low
    "Ni",       # 21 shuddha Ni
)


def _hz_to_note_name(hz: float, tuning: "GlobalTuning | None" = None) -> str:
    """Return a human-readable pitch name for *hz* appropriate to *tuning*.

    - 12-TET / pythagorean / just (12 pitch classes): Western note name
      (C4, A#3, …) — MIDI-based, invariant to tuning.root_hz.
    - 22-śruti custom (Indian ragas): Sargam name (Sa, Re, koGa, Pa, …)
      expressed as śruti degree from tuning.root_hz (Sa).
    - Any other custom N-TET: shows "+N st" step offset from root.
    """
    if hz <= 0.0:
        return "---"

    # ── 12-pitch-class path (12tet, just, pythagorean, or 12-entry custom) ──
    is_12_class = (
        tuning is None
        or tuning.temperament in ("12tet", "just", "pythagorean")
        or (tuning.temperament == "custom" and len(tuning.custom_cents) == 12)
    )
    if is_12_class:
        semitones_from_a4 = 12.0 * math.log2(hz / 440.0)
        midi   = round(semitones_from_a4) + 69   # A4 = MIDI 69
        octave = (midi // 12) - 1
        pc     = midi % 12
        return f"{_NOTE_NAMES_12[pc]}{octave}"

    # ── 22-śruti path ────────────────────────────────────────────────────────
    if tuning.temperament == "custom" and len(tuning.custom_cents) == 22:
        st     = tuning.hz_to_semitones(hz)        # steps in 22-śruti space from root_hz
        idx    = round(st)
        octave = idx // 22
        sruti  = idx % 22
        name   = _SRUTI_NAMES_22[sruti]
        if octave == 0:
            return name
        return f"{name}{'+' if octave > 0 else ''}{octave}oct"

    # ── Generic N-TET fallback ───────────────────────────────────────────────
    dpo = tuning.divisions_per_octave
    st  = tuning.hz_to_semitones(hz)
    idx = round(st)
    octave = idx // dpo
    deg    = idx % dpo
    if octave == 0:
        return f"st{deg}"
    return f"st{deg}{'+' if octave > 0 else ''}{octave}oct"


def _routing_node_label(key: str, patch: "AnalyticPatch") -> str:
    if key == "__patch_tonic__":
        return f"Tonic ({_hz_to_note_name(patch.seq_tonic_hz, patch.tuning)})"
    if key == "__patch_seq__":
        return "Seq.Pitch"
    if key in _DEMO_VIRTUAL_LABELS:
        return _DEMO_VIRTUAL_LABELS[key]
    for i, sys_key in enumerate(_system_input_keys(patch)):
        if sys_key == key:
            return f"System In {i + 1}"
    for i, sys_key in enumerate(_system_output_keys(patch)):
        if sys_key == key:
            return f"System Out {i + 1}"
    if key == "__mix__":
        return "Mix"
    for m in patch.mixers:
        if m.key == key:
            return m.label
    for v in patch.voices:
        if v.key == key:
            return v.label
    for l in patch.lfos:
        if l.key == key:
            return l.label
    for mod in patch.modules:
        if mod.key == key:
            return mod.label
        if mod.module_type == "interaural":
            if mod.ch1_key() == key:
                return f"{mod.label} ch1"
            if mod.ch2_key() == key:
                return f"{mod.label} ch2"
        elif mod.module_type == "lfo" and mod.lfo_channels:
            for i in range(len(mod.lfo_channels)):
                if mod.lfo_ch_key(i) == key:
                    return f"{mod.label} ch{i}"
        elif mod.module_type == "state_machine":
            for item in mod.sm_items:
                for var in mod.sm_vars:
                    if mod.sm_out_key(item, var) == key:
                        return f"{mod.label}.{item}.{var}"
    for cs in patch.controls:
        for sl in cs.sliders:
            if sl.key == key:
                return f"{cs.label}/{sl.label}"
    for pn in patch.param_nodes:
        if pn.key == key:
            return pn.label
    return key[:4]


def _routing_node_color(key: str, patch: "AnalyticPatch") -> tuple:
    if key in _DEMO_VIRTUAL_COLORS:
        return _DEMO_VIRTUAL_COLORS[key]
    if key in _system_input_keys(patch):
        return (70, 170, 170)
    if key in _system_output_keys(patch):
        return (220, 170, 90)
    if key == "__mix__":
        return (200, 200, 100)
    for m in patch.mixers:
        if m.key == key:
            return tuple(m.color[:3])
    for v in patch.voices:
        if v.key == key:
            return tuple(v.color[:3])
    for l in patch.lfos:
        if l.key == key:
            return tuple(l.color[:3])
    for mod in patch.modules:
        if mod.key == key:
            return tuple(mod.color[:3])
        if mod.module_type == "interaural" and key in (mod.ch1_key(), mod.ch2_key()):
            return tuple(mod.color[:3])
        if mod.module_type == "lfo" and mod.lfo_channels:
            for i in range(len(mod.lfo_channels)):
                if mod.lfo_ch_key(i) == key:
                    return tuple(mod.color[:3])
        if mod.module_type == "state_machine" and key in mod.sm_out_keys():
            return tuple(mod.color[:3])
    for cs in patch.controls:
        for sl in cs.sliders:
            if sl.key == key:
                return tuple(cs.color[:3])
    for pn in patch.param_nodes:
        if pn.key == key:
            return tuple(pn.color[:3])
    return (120, 120, 120)
def _working_routing_graph_for_synthesis(patch: "AnalyticPatch") -> RoutingGraph:
    """Return a non-mutating routing graph for preview / render solves.

    Legacy patches with an entirely empty signal-routing graph still need a
    default source->mix path so they remain audible. Once the user has created
    any explicit signal edges, missing edges stay missing; deleted routes are
    not silently reintroduced during preview or rendering.
    """
    if getattr(patch, "routers", None):
        g = RoutingGraph()
        base = RoutingGraph.from_dict(patch.routing.to_dict())
        for nk in base.node_keys():
            g.add_node(
                nk,
                node_type=base.get_node_type(nk),
                source_router_types=(
                    base.get_node_source_router_types(nk)
                    if nk in base.node_source_router_types else None
                ),
                sink_router_types=(
                    base.get_node_sink_router_types(nk)
                    if nk in base.node_sink_router_types else None
                ),
            )
        g.edges.extend(copy.copy(e) for e in base.edges)
        g.param_edges.extend(copy.copy(pe) for pe in base.param_edges)
        g.meta_edges.extend(copy.copy(me) for me in getattr(base, "meta_edges", []))
        g.feedback = copy.deepcopy(base.feedback)
        g.latency_compensation = bool(getattr(base, "latency_compensation", False))
        # New-model patches: merge deployed router graphs into one synthesis
        # graph so the render path can consume the same router instances the
        # editor exposes.  Feedback policy remains graph-global for now.
        for ri, router in enumerate(patch.routers):
            rg = RoutingGraph.from_dict(router.graph.to_dict())
            if ri == 0 and not base.edges and not base.param_edges:
                g.feedback = copy.deepcopy(rg.feedback)
                g.latency_compensation = bool(getattr(rg, "latency_compensation", False))
            for nk in rg.node_keys():
                g.add_node(
                    nk,
                    node_type=rg.get_node_type(nk),
                    source_router_types=(
                        rg.get_node_source_router_types(nk)
                        if nk in rg.node_source_router_types else None
                    ),
                    sink_router_types=(
                        rg.get_node_sink_router_types(nk)
                        if nk in rg.node_sink_router_types else None
                    ),
                )
            g.edges.extend(copy.copy(e) for e in rg.edges)
            g.param_edges.extend(copy.copy(pe) for pe in rg.param_edges)
            g.meta_edges.extend(copy.copy(me) for me in getattr(rg, "meta_edges", []))
    else:
        g = RoutingGraph.from_dict(patch.routing.to_dict())
    mixer_keys = [m.key for m in patch.mixers]
    default_mix_key = mixer_keys[0] if mixer_keys else "__mix__"
    auto_signal_keys = _auto_mix_signal_keys(patch)
    if not g.edges:
        g.ensure_defaults(auto_signal_keys, mix_key=default_mix_key)
    _sanitize_system_io_edges(g, patch)
    g.prune()
    return g
