from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub(name: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m


for _dep in [
    "pygame", "pygame.locals", "pygame.mixer", "pygame.font",
    "pygame.draw", "pygame.transform", "pygame.display",
    "scipy", "scipy.signal",
    "OpenGL", "OpenGL.GL", "OpenGL.GL.shaders",
    "bass_viewer", "plot_widget",
]:
    _stub(_dep)

_pg = sys.modules["pygame"]
for _name in ["KMOD_CTRL", "KMOD_SHIFT"]:
    setattr(_pg, _name, 0)
_pgl = sys.modules["pygame.locals"]
for _name in [
    "DOUBLEBUF", "KEYDOWN", "MOUSEBUTTONDOWN", "MOUSEBUTTONUP", "MOUSEMOTION",
    "MOUSEWHEEL", "OPENGL", "QUIT", "RESIZABLE", "VIDEORESIZE", "K_SPACE",
    "K_ESCAPE", "K_TAB", "K_DELETE", "K_s", "K_o", "K_n", "K_LCTRL",
    "K_RCTRL", "K_z",
]:
    setattr(_pgl, _name, 0)

_scipy_signal = sys.modules["scipy.signal"]
setattr(_scipy_signal, "resample_poly", lambda x, *_a, **_k: x)

_gl = sys.modules["OpenGL.GL"]
for _name in [
    "GL_BLEND", "GL_CLAMP_TO_EDGE", "GL_COLOR_BUFFER_BIT", "GL_LINEAR",
    "GL_LINE_LOOP", "GL_LINE_STRIP", "GL_LINES", "GL_NEAREST",
    "GL_ONE_MINUS_SRC_ALPHA", "GL_QUADS", "GL_RGBA", "GL_SRC_ALPHA",
    "GL_TEXTURE_2D", "GL_TEXTURE_MAG_FILTER", "GL_TEXTURE_MIN_FILTER",
    "GL_TEXTURE_WRAP_S", "GL_TEXTURE_WRAP_T", "GL_TRIANGLES", "GL_UNSIGNED_BYTE",
    "glBegin", "glBindTexture", "glBlendFunc", "glClear", "glClearColor",
    "glColor4f", "glDeleteTextures", "glDisable", "glEnable", "glEnd",
    "glGenTextures", "glLineWidth", "glTexCoord2f", "glTexImage2D",
    "glTexParameteri", "glVertex2f", "glViewport",
]:
    if not hasattr(_gl, _name):
        setattr(_gl, _name, 0 if _name.startswith("GL_") else (lambda *a, **k: None))

_plot_widget = sys.modules["plot_widget"]
for _name in ["PlotWidget", "PlotSeries", "PlotMarker"]:
    setattr(_plot_widget, _name, type(_name, (), {}))

_bass_viewer = sys.modules["bass_viewer"]
for _name in [
    "GlyphAtlas", "Panel", "PanelDock",
    "ScrollableSubpanelList", "ModularSubpanelSpec", "SubpanelAddOption",
]:
    setattr(_bass_viewer, _name, type(_name, (), {}))

from analytic_driver import (
    AnalyticPatch,
    AnalyticVoice,
    AnalyticModule,
    PortTensorSpec,
    PublishedPort,
    ParamNode,
    _control_add_connection,
    _control_remove_connections_for_port,
    _ports_compatible,
    _published_port_lookup,
    _active_routing_graph_and_instance,
    _published_ports_for_patch,
    _rack_device_views_for_patch,
    _router_key_from_ui_key,
    _router_ui_key,
    _routing_grid_node_keys,
    _working_routing_graph_for_synthesis,
)
from routing_engine import MetaEdge, ParamEdge, RouterInstance, RoutingEdge


def test_router_ui_key_roundtrip():
    ui_key = _router_ui_key("vr_main")
    assert ui_key == "__router__:vr_main"
    assert _router_key_from_ui_key(ui_key) == "vr_main"


def test_active_routing_graph_returns_selected_router_instance():
    patch = AnalyticPatch()
    router = RouterInstance(key="vr1", router_type="voice_router", label="Voice Main")
    router.graph.add_node("voice_a")
    patch.routers.append(router)

    graph, active = _active_routing_graph_and_instance(patch, _router_ui_key("vr1"))

    assert active is router
    assert graph is router.graph


def test_routing_grid_node_keys_include_router_filtered_graph_nodes():
    patch = AnalyticPatch()
    voice = AnalyticVoice()
    voice.key = "voice_a"
    voice.label = "Voice A"
    patch.voices.append(voice)

    router = RouterInstance(key="vr1", router_type="voice_router", label="Voice Main")
    router.graph.add_node("voice_a", source_router_types=["voice_router"], sink_router_types=[])
    router.graph.add_node("driver_a", source_router_types=["driver_router"], sink_router_types=["voice_router"])
    router.graph.add_node("master_a", source_router_types=["master"], sink_router_types=["instrument"])
    router.graph.edges.append(RoutingEdge("voice_a", "driver_a", 1.0, router_key="vr1"))
    router.graph.edges.append(RoutingEdge("driver_a", "master_a", 1.0, router_key="master1"))
    patch.routers.append(router)

    keys = _routing_grid_node_keys(patch, _router_ui_key("vr1"))

    assert "voice_a" in keys
    assert "driver_a" in keys
    assert "master_a" not in keys


def test_working_routing_graph_merges_router_instances_for_synthesis():
    patch = AnalyticPatch()

    r1 = RouterInstance(key="vr1", router_type="voice_router", label="Voice Main")
    r1.graph.add_node("voice_a")
    r1.graph.add_node("driver_a")
    r1.graph.edges.append(RoutingEdge("voice_a", "driver_a", 1.0, router_key="vr1"))

    r2 = RouterInstance(key="master1", router_type="master", label="Master")
    r2.graph.add_node("driver_a")
    r2.graph.add_node("master_a")
    r2.graph.edges.append(RoutingEdge("driver_a", "master_a", 0.5, router_key="master1"))

    patch.routers = [r1, r2]

    g = _working_routing_graph_for_synthesis(patch)

    assert "voice_a" in g.node_keys()
    assert "driver_a" in g.node_keys()
    assert "master_a" in g.node_keys()
    assert any(e.src_key == "voice_a" and e.dst_key == "driver_a" for e in g.edges)
    assert any(e.src_key == "driver_a" and e.dst_key == "master_a" for e in g.edges)


def test_published_ports_include_param_targets_and_router_ports():
    patch = AnalyticPatch()
    voice = AnalyticVoice()
    voice.key = "voice_a"
    voice.label = "Voice A"
    patch.voices.append(voice)
    router = RouterInstance(key="vr1", router_type="voice_router", label="Voice Main")
    patch.routers.append(router)

    published = _published_ports_for_patch(patch)

    voice_ports = published["voice_a"]
    assert any(p.domain == "signal" and p.key == "voice_a" for p in voice_ports)
    assert any(p.domain == "param_target" and p.param_path == "chirp.f_delta_start" for p in voice_ports)

    router_ports = published[_router_ui_key("vr1")]
    assert any(p.param_path == "feedback.decay" for p in router_ports)


def test_state_machine_module_publishes_control_ports_with_parallel_group():
    patch = AnalyticPatch()
    mod = AnalyticModule()
    mod.key = "sm1"
    mod.label = "Performer SM"
    mod.module_type = "state_machine"
    mod.sm_n_items = 4
    mod.sm_items = ["chair_a", "chair_b"]
    mod.sm_vars = ["feedback_pressure", "aperture_pressure"]
    patch.modules.append(mod)

    published = _published_ports_for_patch(patch)
    ports = published["sm1"]

    ctrl_ports = [p for p in ports if p.domain == "control"]
    assert ctrl_ports
    assert all(p.tensor.parallel_group == "sm_ctrl:sm1" for p in ctrl_ports)
    assert all(p.tensor.batchable for p in ctrl_ports)


def test_state_machine_module_publishes_declared_bundle_ports():
    patch = AnalyticPatch()
    mod = AnalyticModule()
    mod.key = "score_sm"
    mod.label = "Score SM"
    mod.module_type = "state_machine"
    mod.sm_bundle_ports = [{
        "name": "score",
        "direction": "out",
        "domain": "control",
        "semantic_role": "score_bundle",
        "tensor_rank": 3,
        "lane_count": 0,
        "batch_axes": 2,
        "parallel_group": "score_bundle",
        "group_validity": "remap",
        "channel_dims": [16, 4],
    }]
    patch.modules.append(mod)

    published = _published_ports_for_patch(patch)
    ports = published["score_sm"]
    score_port = next(p for p in ports if p.key == "score_sm:score")

    assert score_port.semantic_role == "score_bundle"
    assert score_port.tensor.channel_dims == [16, 4]
    assert score_port.tensor.group_validity == "remap"


def test_rack_device_views_pack_published_devices():
    patch = AnalyticPatch()
    v1 = AnalyticVoice()
    v1.key = "voice_a"
    patch.voices.append(v1)
    pn = ParamNode()
    pn.key = "param_a"
    pn.targets = [{"voice_key": "voice_a", "attr": "amplitude"}]
    patch.param_nodes.append(pn)

    rack = _rack_device_views_for_patch(patch)

    keys = {d.device_key for d in rack}
    assert "voice_a" in keys
    assert "param_a" in keys
    assert all(d.rack_u >= 1 and d.rack_w >= 1 for d in rack)
    assert any(d.ports for d in rack)


def test_control_add_connection_writes_param_edge_with_ports():
    patch = AnalyticPatch()
    voice = AnalyticVoice()
    voice.key = "voice_a"
    patch.voices.append(voice)
    mod = AnalyticModule()
    mod.key = "sm1"
    mod.label = "SM"
    mod.module_type = "state_machine"
    mod.sm_n_items = 2
    mod.sm_items = ["a"]
    mod.sm_vars = ["feedback_pressure"]
    patch.modules.append(mod)

    ports = _published_port_lookup(patch)
    src = next(p for p in ports.values() if p.key == "sm1.ctrl.a.feedback_pressure")
    dst = next(p for p in ports.values() if p.key == "voice_a:param:amplitude")

    assert _control_add_connection(patch, src, dst) is True
    assert len(patch.routing.param_edges) == 1
    pe = patch.routing.param_edges[0]
    assert pe.src_port == src.key
    assert pe.dst_port == dst.key
    assert pe.tensor_contract.analytic_only is True
    assert pe.projection_policy == "magnitude_mean"
    assert pe.transfer_spec.transfer_policy == "remap"
    assert pe.transfer_spec.batch_policy == "strict"
    assert pe.transfer_spec.lane_policy in {"strict", "remap"}
    assert pe.transfer_spec.reduction == ""


def test_ports_compatible_reject_mismatched_parallel_groups_without_negotiation():
    src = PublishedPort(
        key="room:out",
        direction="out",
        domain="control",
        owner_key="room",
        tensor=PortTensorSpec(
            tensor_rank=2,
            lane_count=8,
            batch_axes=2,
            parallel_group="room_feedback",
            group_validity="strict",
        ),
    )
    dst = PublishedPort(
        key="performer:in",
        direction="in",
        domain="control",
        owner_key="performer",
        tensor=PortTensorSpec(
            tensor_rank=2,
            lane_count=8,
            batch_axes=2,
            parallel_group="performer_feedback",
            group_validity="strict",
        ),
    )

    assert _ports_compatible(src, dst) is False


def test_ports_compatible_accept_group_remap_when_destination_negotiates():
    src = PublishedPort(
        key="room:out",
        direction="out",
        domain="control",
        owner_key="room",
        tensor=PortTensorSpec(
            tensor_rank=2,
            lane_count=8,
            batch_axes=2,
            parallel_group="room_feedback",
            group_validity="remap",
        ),
    )
    dst = PublishedPort(
        key="performer:in",
        direction="in",
        domain="control",
        owner_key="performer",
        tensor=PortTensorSpec(
            tensor_rank=2,
            lane_count=8,
            batch_axes=1,
            parallel_group="performer_feedback",
            group_validity="strict",
        ),
        negotiates_group_validity=True,
    )

    assert _ports_compatible(src, dst) is True


def test_control_remove_connections_for_port_prunes_incident_edges():
    patch = AnalyticPatch()
    patch.routing.param_edges = []
    patch.routing.edges = []
    patch.routing.edges.append(RoutingEdge(
        src_key="a", dst_key="b", edge_kind="control", src_port="a:out", dst_port="b:in"
    ))
    patch.routing.param_edges.append(ParamEdge(
        src_key="a:out", dst_key="voice_a", param_path="amplitude",
        src_port="a:out", dst_port="voice_a:param:amplitude"
    ))
    patch.routing.meta_edges.append(MetaEdge(
        a_key="sm_a", b_key="sm_b", a_port="a:out", b_port="b:in"
    ))

    _control_remove_connections_for_port(patch, "a:out")

    assert patch.routing.edges == []
    assert patch.routing.param_edges == []
    assert patch.routing.meta_edges == []


def test_state_machine_to_state_machine_connection_authors_metaedge():
    patch = AnalyticPatch()

    score = AnalyticModule()
    score.key = "score_sm"
    score.label = "Score SM"
    score.module_type = "state_machine"
    score.sm_bundle_ports = [{
        "name": "score",
        "direction": "out",
        "domain": "control",
        "semantic_role": "score_bundle",
        "tensor_rank": 3,
        "lane_count": 0,
        "batch_axes": 2,
        "parallel_group": "score_bundle",
        "group_validity": "remap",
        "channel_dims": [16, 4],
    }]

    performer = AnalyticModule()
    performer.key = "performer_sm"
    performer.label = "Performer SM"
    performer.module_type = "state_machine"
    performer.sm_bundle_ports = [{
        "name": "score",
        "direction": "in",
        "domain": "control",
        "semantic_role": "score_bundle",
        "tensor_rank": 3,
        "lane_count": 0,
        "batch_axes": 2,
        "parallel_group": "performer_score",
        "group_validity": "remap",
        "channel_dims": [16, 4],
    }]

    patch.modules.extend([score, performer])
    ports = _published_port_lookup(patch)
    src = ports["score_sm:score"]
    dst = ports["performer_sm:score"]

    assert _control_add_connection(patch, src, dst) is True
    assert patch.routing.edges == []
    assert patch.routing.param_edges == []
    assert len(patch.routing.meta_edges) == 1
    me = patch.routing.meta_edges[0]
    assert me.a_port == "score_sm:score"
    assert me.b_port == "performer_sm:score"
    assert me.semantic_role == "score_bundle"
    assert me.tensor_contract.channel_dims == [16, 4]
    assert me.a_to_b_transfer.transfer_policy in {"broadcast", "remap", "identity"}
