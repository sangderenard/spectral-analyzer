# Native repository-world windows

`native_repo_world_islands.RepoWorldIslands` adapts Turing's existing
`RepoWorldScan -> UnitIndex -> build_unit_world` pipeline. It does not scan
new files, compile a runtime, or invent another map document schema.

```python
from src.compiler.repo_world_units import UnitIndex
from native_repo_world_islands import RepoWorldIslands
from nodus_console_station import workspace_from_graph

index = UnitIndex(authorized_scan, capacity=32)
worlds = RepoWorldIslands(index, lambda geometry: workspace_from_graph(
    {'modules': [], 'edges': [], 'connectors': []}, geometry=geometry), scale=4)
loader = worlds.loader(initial_unit_address)
```

Run that loader directly through the existing application, with normal Pluck
CLI settings (do not also select another initial world):

```python
import demo_pluck_gl
demo_pluck_gl.main(island_loader=loader)
```

The same actual GL main-loop attachment handles procedural worlds. Register
no executable metadata from documents. The scan/index is explicitly supplied
by the host with its existing authorization, not inferred from a directory
name or fetched from a door's href.

A treadmill event selects the original `Treadmill.address(offset)`. Only that
contiguous window is materialized, including partial windows at the tail.
The original producer's `href` callback carries a UnitIndex address to identify
the correct parent window on return. It is interpreted only as an index address;
it is never opened as a file or URL. Cached windows preserve workspace edits.

A standalone `abstract-ui-div-map-v0` JSON document contains only its realized
window and cannot reconstruct unseen repository items. Consequently the static
`--living-map-islands` manifest still refuses treadmill events. It needs this
explicit index-backed host factory, not a synthetic window or guessed source.

Verification: real `scan_files` input, `UnitIndex(capacity=2)`, five-item child
entered at offset 4 produces one visible item; return reaches the original
parent window and the same edited workspace object. No native GPU test or new
collision solver is implied by this producer test.
