# Camera trigger bus: existing graph and mixer

Source-only integration seam. No live server or GPU renderer was changed.

## What is implemented

`camera_software/trigger_bus.py` consumes the existing `ExposureFrame`, preserving
the exact object, original slices, scene times and `CameraCausalityKey` identities.
It registers ordinary `RoutingGraph` nodes and uses the original
`network_materializer.routing_edge_to_tensor_edge` and `compile_nodes` entry to
execute `GraphSolver`. There is no second scheduler or graph solver.

Three camera-authored sources are available:

- `Sync`: one at the first slice, zero afterward. A frame-start trigger.
- `Flash`: each slice's existing `flash_weight`, unchanged.
- `Gate`: one while the existing slice says `sensor_integrating`, zero otherwise.

Named wire buses and exact fixture/receiver IDs are ordinary nodes. They start
unpatched. Gain, phase, fan-out, fan-in and algebraic loops use the original
complex128 solver. Results are **control signals**, not new radiant power, photons
or a declaration that a physical flash fired. A light consumer must interpret the
chosen source, retain fixture identity and apply energy weighting exactly once.

The UI is the actual `analytic_driver.RoutingGridView` and its original event
handlers. A named `RouterInstance` exposes the same graph without synthetic audio
voices. No UI implementation was copied or added. Normal `RoutingGraph.to_dict` /
`from_dict` is the storage contract. Example:

```python
from camera_software.trigger_bus import (
    make_camera_trigger_graph, make_camera_trigger_panel, resolve_camera_trigger_bus,
)

graph = make_camera_trigger_graph(["Key", "Fill", "Rim"], bus_keys=["Bus"])
graph.set_weight("Flash", "Bus", 1.0)
graph.set_weight("Bus", "Key", 1.0)
graph.set_weight("Bus", "Fill", 0.5)
patch, view = make_camera_trigger_panel(graph)
surface = view.render_surface(patch, 1000, 620)
# Forward normal Pygame events to view.handle_event(event, patch, x, y).
result = resolve_camera_trigger_bus(existing_exposure_frame, graph, ["Key", "Fill", "Rim"])
```

## Boundaries that remain

The Studio consumer now admits finite nonnegative delays on the original graph's
uniform 0.1 ms control grid, limited to a five-second pulse and delayed tail.
It rejects variable-length delayed slices, delayed cycles, Sync/Gate-sourced
illumination, parameter/meta edges, nonlinear saturation and wide transfers.
The original mixer still shows its full controls; the host surfaces explicit
admission errors for settings outside this camera-light contract. The same graph
also drives a manual Flash pulse in the Phong preview through a browser image
adapter for the original Pygame `RoutingGridView`.

Three optional `mod:1`–`mod:3` source nodes are camera-triggered modulators.
Their samples come from `signal_generator_v2`'s `ConstantPhasePath`, sine/warped/
harmonic manifolds, linear/piecewise ramps, and `ADSREnvelope`. Each Flash rising
edge starts the selected waveform; repeat count and spacing restart it within
the same camera event. Attack, decay, sustain level, sustain time and release are
explicit controls. The resulting arrays enter the same `GraphSolver` as Flash
and are patched to named light or rack receivers through the original mixer.
Generator settings are a separate bounded camera contract alongside the saved
`RoutingGraph`; they do not replace graph gain, phase, delay or fan-out.

No existing `peanut` / optical-slave receiver was found in the checked camera or
signal-graph source. If peanut means an optical slave, connect an actual detected
light signal and its threshold/rearm/latency behavior; do not relabel an electrical
wire as optical detection. That receiver is not implemented here.

Studio `prepare_photo_geometry` already places native camera housing using the
capture eye/target. Its recorded geometry version is
`native-housing-behind-thin-lens-v1`; optics are still analytical thin lens.
The thick-lens lab separately uses `SceneCameraCoordinator`, `CameraTimeline` and
`ExposureBarrier`. Studio does not yet run that whole camera program, and this
adapter does not pretend to install refractive lens surfaces or run fixture
emission. The mixer is mounted in Studio's browser through the original Pygame
event handlers. The browser relays pointer events and receives a PNG and the
original `RoutingGraph` serialization; it does not own a second graph solver.

`seconds` remains the Studio worker's wall-clock compute budget. Photographic
time belongs to `camera_light_burst.exposure_time_s`, already authoritative in
`CameraExposureScheduler`. ISO still needs a calibrated sensor/back path; it must
not be confused with display gain or with more Monte Carlo samples.

## Verification

- Baseline existing exposure, graph-delay and routing-default suites: 50 passed.
- Same suites plus `tests/test_camera_trigger_bus.py`: 65 passed.
- `python tests/camera_trigger_bus_panel_check.py`: actual CPU Pygame surface and
  event test, outside the test suite's GUI stubs. It checks disconnect, repaint
  without reconnecting, and double-click reconnect through the original mixer,
  with each change measured through the real GraphSolver.
- No OpenGL context, production restart, web export or active exposure changes.

The panel check writes `build/camera-trigger-bus-panel.png`. These are bounded
component checks, not an integrated optical render or a full-repository gate.
