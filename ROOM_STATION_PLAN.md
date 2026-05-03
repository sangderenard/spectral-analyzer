# Room Station — Action Plan

## Overview

`RoomStation` is the master duty station that owns and controls the
physical environment the player inhabits.  All other duty stations
(Fabricator, Simulator, …) are *guests* placed inside the room.

The "branch point" from `demo_pluck_gl.py` is `room_demo.py` — a
self-contained entry point that runs the duty-station-only system
without the guitar physics.

---

## Architecture Diagram

```
room_demo.py (entry point)
  ├── RoomWorkspace          pure state machine; owns scene registry
  │     ├── PlacedLight × N
  │     ├── PlacedCamera × N
  │     ├── PlacedEnclosure × N   (rect | cyl | sphere | tablet_rect | tablet_polar)
  │     ├── PlacedDutyStation × N  (fabricator | simulator | …)
  │     └── PlacedPortalFrame × N  (stub — not yet functional)
  │
  ├── RoomStation            GL renderer  +  room-control HUD
  │     ├── _SceneTreePanel  (left)  scrollable object list
  │     ├── _MinimapPanel    (centre) top-down floor plan
  │     └── _PropertiesPanel (right) selected-object fields
  │
  ├── room_geometry.py       builds floor / wall / ceiling VAO
  ├── enclosure_geometry.py  all enclosure shapes (see below)
  ├── glass_room.py          Enclosure class dispatches to enclosure_geometry
  │
  ├── FabricatorStation / FabricatorWorkspace   (existing)
  ├── SimulatorStation / SimulatorWorkspace      (this session)
  └── PlayerController       physics params come from RoomWorkspace
```

---

## Enclosure Shapes (enclosure_geometry.py)

| shape_type      | Geometry                                | Pedestal |
|-----------------|------------------------------------------|----------|
| `rect`          | 4 flat walls + rounded corners + top cap | opaque skirt |
| `cyl`           | cylindrical shell + top annular cap      | cylindrical skirt |
| `sphere`        | UV sphere shell (full)                   | cylindrical pedestal |
| `tablet_rect`   | two parallel rectangular glass panes     | none (floating) |
| `tablet_polar`  | two parallel circular glass disk panes   | none (floating) |

All shapes return **float32 (-1, 6)** `[x, y, z, nx, ny, nz]` for the
opaque Phong pass and **float32 (-1, 3)** `[x, y, z]` for the
wireframe/line pass.

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `ROOM_STATION_PLAN.md` | ✅ | this document |
| `configs/room_station/room.yaml` | → | room dimensions + materials |
| `configs/room_station/physics.yaml` | → | walk speed, gravity, eye height |
| `configs/room_station/station.yaml` | → | room-station console position + UI layout |
| `configs/room_station/scene.yaml` | → | initial placed-object list |
| `placed_object.py` | → | data-classes: PlacedLight, PlacedCamera, PlacedEnclosure, … |
| `enclosure_geometry.py` | → | geometry builders for all enclosure shapes |
| `room_geometry.py` | → | box-room mesh (floor, walls, ceiling) |
| `room_workspace.py` | → | pure state machine; scene registry + physics params |
| `room_station.py` | → | GL renderer; minimap + scene tree + properties panels |
| `room_demo.py` | → | standalone entry point (branch from demo_pluck_gl) |
| `glass_room.py` | → | add `Enclosure` class dispatching to enclosure_geometry |
| `fabricator_workspace.py` | → | add `light_point`, `light_spot`, `portal_frame` to catalog |

---

## Initial Scene Layout (scene.yaml)

A 12 × 4 × 10 m room (x × y × z) with:

```
                +z (depth)
     ┌──────────────────────────┐
     │  [fab]         [portal]  │ z=8
     │                          │
     │  [cyl sim]   [sph sim]   │ z=4
     │                          │
     │  [rect sim]  [tablets]   │ z=2
     │                          │
     │       [console]          │ z=0
     └──────────────────────────┘
           x=-5   x=0   x=5
```

Objects:
- 3× enclosures (rect / cyl / sphere) — each wraps a SimulatorWorkspace
  with different grid sizes (4, 9, 16 items)
- 2× floating glass tablets (rect + polar)
- 1× FabricatorStation console
- 3× point lights (warm, on ceiling)
- 1× portal frame (stub)
- Room-station console at [0, 1.0, 0.3] (near entrance)

---

## Physics Configuration

`RoomWorkspace.physics_config()` returns a dict wired into
`PlayerController` at construction time:

```yaml
walk:
  start_pos:      [0.0,  0.0,  0.8]   # near entrance
  start_yaw:      0.0
  eye_height:     1.65
  move_speed:     3.0
  run_multiplier: 2.2
  mouse_sensitivity: 0.15
  pitch_limit_deg:   75.0
```

`RoomWorkspace` also exposes `apply_physics(player_ctrl)` so that
in-room edits (e.g. lowering gravity via the properties panel) update
the live player controller.

---

## Fabricator Catalog Extensions

Two new special-item types added to `FabricatorWorkspace`:

| id | label | kind | behavior |
|----|-------|------|----------|
| `light_point` | Point Light | light | places a PlacedLight in room registry |
| `light_spot`  | Spot Light  | light | places a PlacedLight (spot) |
| `portal_frame`| Portal Frame| portal | stub — places PlacedPortalFrame |

These appear in the fabricator's left-panel palette below the solids list.
`FabricatorWorkspace.catalog_items()` returns `list[dict]` with all
palette entries (solids + special items).

---

## Deferred / Out-of-Scope

- Portal traversal logic (PlacedPortalFrame is a visual stub only)
- Runtime light/shadow casting (lights are scene-graph data; rendering
  uses a fixed sun light for now)
- Network multiplayer / multi-room navigation
- Terrain / irregular floor geometry
