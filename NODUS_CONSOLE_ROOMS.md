# Native Nodus console rooms

The existing Pluck first-person application now accepts:

    python demo_pluck_gl.py --nodus-graph <authorized-GRAPH-export.json>

Optional `--nodus-consoles <selection.json>` chooses arbitrary module sets and
supplied Pluck Panel records. Each selection is keyed by console identity:

    {"mixer":{"module_ids":["17","23"],"label":"Mixer","panel":{"name":"mixer","label":"Mixer","knobs":[],"payload":{"actions":[{"key":"apply","label":"Apply"}]}}}}

No server connection or credentials are synthesized. The normal Pluck
RoomWorkspace and DutyStation build/render path is reused. The factory
`nodus_console_station.workspace_from_graph` is also available for an existing
host; `station.menu` exposes the actual NodusConsoleMenu.

Hosts can bind an authorized action handler with `menu.bind_host(dispatch=...,
subscribe=...)`. Actions run through Pluck's existing ActionRegistry/queue.
The callback receives module IDs, action route and current values; the host
must enforce actual account/row authority. Serialized scene metadata never
contains executable callbacks. Without a handler, actions are read-only.

Push true graph-owner snapshots using `update_stream(kind,data)` for metadata,
api, abi, profile or stats. Opt-in toggles call the host subscription handler;
no fake values or automatic high-rate polling is supplied. `update_values`
updates rendered setting values. Each host reads Nodus on its graph stepping
thread; render-thread code only reads copied snapshots. Next/previous panel
controls expose nested authored panels and enabled feeds.

Verified: real workspace→DutyStation→NodusConsoleMenu construction, panel and
stream behavior tests, Python compilation and diff hygiene. No GPU window or
live Nodus service was opened. Actual visible knobs' editing interaction,
authenticated remote bindings and browser/native visual acceptance remain.
Native portal traversal and continuous scroll worlds are not invented here;
Turing's separate graph-world adapter uses its existing paired-door runtime.

## Live read console and settings interaction

The normal app also accepts `--nodus-console-port PORT
--nodus-console-token-file FILE`, using the existing authenticated loopback
Nodus console. Make the existing `nodus_graph` package available on PYTHONPATH.
The file is read only when explicitly selected; its value is never printed or
passed on the command line. No remote/plaintext Internet endpoint is enabled.

The stream buttons now subscribe to real GRAPH/TOOLS/ACTIVITY reads using
NodusConsoleFeeds; stats/API carry canvas scope because the source protocol
has no narrower row metrics. Profile explicitly reports unavailable because
that protocol has no profile verb. Graph-owner snapshots are not read from the
rendering thread. Stop/reconnect must preserve client ownership.

Authored stepper/toggle/choice knobs now accept real UI clicks, using the same
shared knob logic as RoomControlStation. An authorized host callback receives
`setting.<knob-name>` through the existing action registry. The read-console
connection intentionally supplies no write handler: its actual protocol is
read-only. Native ownership/REST mutation needs an actual authorized endpoint,
not a newly invented write verb.

PlayerController now accepts the native declared-door runtime through
`set_door_runtime`. Actual walk ticks cross paired doors and update position,
vertical velocity and facing, using explicit Z-up aperture geometry; island
crossing delegates to the supplied host loader. NativeDoorRuntime implements
the existing dial/treadmill declaration selection and preserves its event
contract. Complete visual/world-load acceptance is still pending.

## Native geometry integration

The application now calls the original Turing world/layout producers, without
HTML/WASM compilation, and explicitly converts Y-up to native Pluck Z-up.
`--nodus-world-scale` (default 4) uniformly enlarges the quarter-scale Living
Map environment for the full-size Pluck walker. Console locations, door frames
and traversal apertures share this same transform. Door frames have real
width/height, use existing DutyStation mesh/material submission, and remain
open in the middle. PlayerController receives the generated paired-door runtime.
Original dial/treadmill selectors are available as console destination pages.

Verified geometry-only acceptance: actual producer produces four openings,
two 144-vertex portal frames and one real console for a one-table graph.
Sixteen focused tests pass, including shared knob logic, feed opt-in,
paired-door/cooldown/scroll contracts and real PlayerController Z-up transfer.
No GUI/GPU run or live-server connection has been claimed.

Remaining: remote parameter writes (current console protocol is read-only),
procedural treadmill window factories, collision integration and visual acceptance.
The implementation below supersedes the earlier geometry/loader limitations.

## Native room shells and safe island preparation

Living Map box boundaries now use the existing Pluck DutyStation mesh path.
The adapter follows the original `wallSegments` interval subtraction for
floors, wall sections, sills and lintels; apertures are not sealed by a box.
The original yaw, elevation, dimensions and uniform display scale are retained.
Other geometry modes are not misrepresented as rectangular room shells.
This is visual shell geometry, not a new physics collision implementation.

NativeIslandLoader prepares the declared target arrival before committing a
world switch. Explicit host factories supply approved worlds; descriptive hrefs
are never fetched. Cached workspaces retain controls and edits. Unsupported
or stale destinations fail without replacing the active world.

Derived Nodus/Living Map workspaces cannot implicitly save over the shared
config scene.yaml: saving requires an explicit destination filename. Loading
and projection only read config defaults and maintain their own in-memory state.

Verification: 22 focused tests including actual PlayerController transfer,
room aperture occupancy, coincident portal cuts, source scale, transactional
island refusal/cache preservation, and no-default-save. Actual producer to
native station construction produces two portal frames, six room shell meshes,
and one console for a one-table graph. No GPU or live-service acceptance claimed.


## Run existing Living Map documents and island doors

    python demo_pluck_gl.py --living-map-document outside.json --living-map-islands islands.json

`islands.json` maps original world identities to local document filenames,
resolved relative to the manifest. Each file must be an existing
`abstract-ui-div-map-v0` document with that identity, not executable code.
The manifest may also accompany `--nodus-graph`. Declared hrefs are never
fetched. The main GL thread prepares and caches the target room/stations/cameras,
then replaces only island-owned objects and applies the original arrival pose.
Global stations and the existing authenticated feed session remain intact.
An unregistered or invalid target leaves the current world in place.

The local static-document route deliberately refuses procedural treadmill
window requests. Hosts can register an actual window-aware factory with
NativeIslandLoader; no generated window is silently claimed by this CLI.

Verified: tests execute the actual main-loop handoff block with mocked GL
constructors, confirming successful attachment preserves global stations and
failed preparation preserves the active world. An actual Turing-declared
outside/inside pair completes a native round trip and reuses the original
workspace. This is not yet a visual GPU acceptance run.

## Collision and procedural host integration

Native maps now submit the SAME transformed room-prism triangles to the
existing PlayerClipEngine surface field. Its explicit previous-position mode
uses the existing ray-tracer bridge to sweep the submitted movement and stop
at first triangle contact, instead of the former small proximity nudge. Normal
room clients retain their existing behavior. Map worlds do not inherit the
unrelated four-metre ceiling. Vertical contact clears falling velocity.

This is a center-ray sweep with plane clearance, not a full capsule/manifold
solver or edge-radius sweep. It conservatively stops rather than carrying an
unchecked slide remainder through another wall. Exact wall/opening and rotated
mesh checks pass; GPU/walking acceptance remains outstanding.

Procedural repository windows now use the original UnitIndex and build_unit_world
producer, not a copied layout. An authorized host can pass the resulting loader
to `demo_pluck_gl.main(island_loader=loader)`; see
`docs/NATIVE_REPO_WORLD_WINDOWS.md`. The actual main-loop transition handles it.
A partial tail window and returning to the same edited parent window are verified.
Thirty focused tests pass across collision, geometry, portals, controls and host
attachment. No service, account, session or saved config was changed for testing.

## Shared-edge contract boundary

The opt-in console feeds are read-only views of the existing GRAPH, TOOLS and
ACTIVITY protocol. They neither consume computation tokens nor introduce a
per-tool FIFO. ABI views preserve the graph's original edge/FIFO metadata.
Native Nodus external named ports already use the shared table-edge transport;
future live console actions must use that existing host/edge contract, not a
parallel queue. The Pluck UI ActionRegistry callback is only host dispatch and
is not evidence that remote parameter mutation is implemented.

If GRAPH returns its native-index `definition_error` fallback, the console
marks metadata/ABI unavailable rather than guessing stable identities. Other
feeds continue; the next valid stable graph restores those views. This path
has a regression test, but no live session or execution edge was consumed.
