"""Smoke test for the peer-star naive network."""
from __future__ import annotations

import controls
from cable_object import CableObject
from gateway_object import get_gateway
from naive_graph import (
    Address,
    CableKind,
    NaiveStarDutyStation,
    get_naive_graph_controller,
    make_envelope,
)
from port_plate import PortPlate


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    ctrl = get_naive_graph_controller()

    section("1. Two stars, members + duty stations (DS is a star member)")
    controls.register_input_endpoint(owner_id="A.alpha", endpoint_id="in0", capacity=16)
    controls.register_output_endpoint(owner_id="A.alpha", endpoint_id="out0", capacity=16)
    controls.register_input_endpoint(owner_id="A.beta", endpoint_id="in0", capacity=16)
    controls.register_output_endpoint(owner_id="A.beta", endpoint_id="out0", capacity=16)
    ds_a = NaiveStarDutyStation("A", priority=10, breakout_count=4)
    ctrl.join_star("A.alpha", "A")
    ctrl.join_star("A.beta", "A")

    controls.register_input_endpoint(owner_id="B.gamma", endpoint_id="in0", capacity=16)
    controls.register_output_endpoint(owner_id="B.gamma", endpoint_id="out0", capacity=16)
    ds_b = NaiveStarDutyStation("B", priority=5, breakout_count=4)
    ctrl.join_star("B.gamma", "B")

    print(f"star A members: {ds_a.star.member_ids}")
    print(f"star B members: {ds_b.star.member_ids}")
    print(f"DS A wan: in={ds_a.wan_in_address}, out={ds_a.wan_out_address}")

    section("2. Intra-star LAN delivery (alpha -> beta within A)")
    ctx_alpha = controls.get_control_graph().fifo_context("A.alpha")
    ctx_beta = controls.get_control_graph().fifo_context("A.beta")
    ctx_alpha.output("out0").push(
        make_envelope({"hello": "from-alpha"}, to=Address("A.beta", "in0"))
    )
    print(f"tick: {ctrl.tick()}")
    print(f"beta inbox: {ctx_beta.input('in0').drain()}")

    section("3. Cross-star without cable: stranded at DS A wan_out")
    ctx_alpha.output("out0").push(
        make_envelope({"hello": "to-gamma"}, to=Address("B.gamma", "in0"))
    )
    print(f"tick: {ctrl.tick()}")
    ds_a_wan_out = controls.get_control_graph().fifo_context(ds_a.owner_id).output("wan_out")
    print(f"DS A wan_out pending: {len(ds_a_wan_out)}  (no cable -> stuck)")
    ds_a_wan_out.drain()

    section("4. Plug a NAIVE cable (DS A wan_out -> DS B wan_in) and send across")
    cable = CableObject(kind=CableKind.NAIVE, cable_id="trunk_AB")
    cable.attach_a(ds_a.wan_out_address)
    cable.attach_b(ds_b.wan_in_address)
    cable_back = CableObject(kind=CableKind.NAIVE, cable_id="trunk_BA")
    cable_back.attach_a(ds_b.wan_out_address)
    cable_back.attach_b(ds_a.wan_in_address)
    print(f"trunk_AB plugged: {cable.is_plugged}, trunk_BA plugged: {cable_back.is_plugged}")

    ctx_alpha.output("out0").push(
        make_envelope({"hello": "across"}, to=Address("B.gamma", "in0"))
    )
    # Tick #1: A ticks -> ingests alpha's output, sees gamma is remote, emits via
    # DS A wan_out.  Then cable pump moves it to DS B wan_in.  B ticks AFTER A
    # in this same controller tick (because A.priority=10 > B.priority=5), so
    # B doesn't pick it up yet.
    r1 = ctrl.tick()
    # Tick #2: B ticks -> ingests its WAN-boundary input -> sees gamma is local
    # -> delivers to gamma.in0.
    r2 = ctrl.tick()
    print(f"tick #1: {r1}")
    print(f"tick #2: {r2}")
    print(f"gamma inbox: {controls.get_control_graph().fifo_context('B.gamma').input('in0').drain()}")

    section("5. Loop protection: WAN port already visited cannot re-emit")
    # Inject a hand-crafted envelope into A's inbox with both A's WAN out
    # already visited and a destination only reachable across the WAN.
    msg = make_envelope({"x": 1}, to=Address("B.gamma", "in0"))
    msg.visited_wan.add(ds_a.wan_out_address)
    with ds_a.star._lock:
        ds_a.star._inbox.append(msg)
    r3 = ctrl.tick()
    print(f"tick #3 star A: {r3['stars']['A']}  (loop_blocked should be 1)")

    section("6. Advanced flush plates + class enforcement")
    plate_in = PortPlate(plate_id="adv_in", host_owner_id="A.alpha", direction="input",
                         world_pos=(0.0, 1.0, 0.5))
    plate_out = PortPlate(plate_id="adv_out", host_owner_id="A.beta", direction="output",
                          world_pos=(0.5, 1.0, 0.5))
    plate_in.deploy()
    plate_out.deploy()

    adv = CableObject(kind=CableKind.ADVANCED, cable_id="adv_1")
    adv.attach_a(plate_out.address)
    adv.attach_b(plate_in.address)
    print(f"advanced cable plugged: {adv.is_plugged}")

    bad1 = CableObject(kind=CableKind.ADVANCED, cable_id="bad_1")
    try:
        bad1.attach_a(plate_in.address)
        bad1.attach_b(Address("A.beta", "in0"))
        print("BUG: bad cable accepted!")
        return 1
    except ValueError as exc:
        print(f"OK refusal: {exc}")

    bad2 = CableObject(kind=CableKind.NAIVE, cable_id="bad_2")
    try:
        bad2.attach_a(Address("A.alpha", "out0"))
        bad2.attach_b(plate_in.address)
        print("BUG: bad cable accepted!")
        return 1
    except ValueError as exc:
        print(f"OK refusal: {exc}")

    section("7. Singleton gateway with subscriber")
    received: list = []
    gw = get_gateway(world_pos=(2.0, 0.0, 0.0))
    gw.set_subscriber(received.append)
    controls.get_control_graph().fifo_context(gw.owner_id).input("naive_in").push(
        {"to_gateway": "deliberate-out-of-band"}
    )
    pumped = gw.pump()
    print(f"gateway pumped: {pumped}, received: {received}")
    gw2 = get_gateway()
    print(f"singleton: {gw is gw2}")

    section("8. Duty station + global stats")
    print("DS A:", ds_a.get_stats())
    print("DS B:", ds_b.get_stats())
    print("Global:", ctrl.stats)

    print()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
