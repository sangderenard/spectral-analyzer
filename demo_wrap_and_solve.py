"""demo_wrap_and_solve.py — Load a patch, wrap all analytic objects,
solve one tick with dummy linear identity transforms as defaults.
Also registers every node with DebugArchetype to test the archetype
dispatch path end-to-end.

Usage:
    python demo_wrap_and_solve.py [path/to/patch.json] [--sr 48000] [--ticks 4]

If no patch path is given the script uses the built-in orchestral_placement_test
preset that ships with the repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("patch", nargs="?",
                   default=str(Path(__file__).parent /
                               "presets/patch/orchestral_placement_test.json"),
                   help="Path to a saved patch JSON file")
    p.add_argument("--sr",    type=float, default=48_000.0, help="Sample rate")
    p.add_argument("--ticks", type=int,   default=4,        help="Solver ticks to run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load patch
# ---------------------------------------------------------------------------
def load_patch(path: str):
    import analytic_driver as ad
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return ad.AnalyticPatch.from_dict(d)


# ---------------------------------------------------------------------------
# Dummy identity transform injected for every node that has no transform
# ---------------------------------------------------------------------------
def _identity_transform(x: torch.Tensor) -> torch.Tensor:
    """Pass-through: sum of incoming edges unchanged."""
    return x


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse()

    print(f"[demo] Loading patch: {args.patch}")
    patch = load_patch(args.patch)
    print(f"[demo] Patch loaded: {patch.name!r}")
    print(f"[demo]   voices   : {len(patch.voices)}")
    print(f"[demo]   lfos     : {len(patch.lfos)}")
    print(f"[demo]   modules  : {len(patch.modules)}")
    print(f"[demo]   controls : {sum(len(cs.sliders) for cs in patch.controls)}")
    print(f"[demo]   mixers   : {len(patch.mixers)}")
    print(f"[demo]   param_nodes: {len(patch.param_nodes)}")

    # ── Materialise nodes + edges ────────────────────────────────────────
    print("\n[demo] Materialising network …")
    from network_materializer import materialize_network
    nodes, edges = materialize_network(patch, sr=args.sr)
    print(f"[demo]   TensorNodes : {len(nodes)}")
    print(f"[demo]   TensorEdges : {len(edges)}")

    # Tally which nodes have analytic wrappers vs. which are virtual/passthrough
    wrapped   = [n for n in nodes if n.analytic_module is not None]
    bare      = [n for n in nodes if n.analytic_module is None and n.transform is None]
    has_xform = [n for n in nodes if n.transform is not None]
    print(f"[demo]   analytic wrappers  : {len(wrapped)}")
    print(f"[demo]   passthrough (bare) : {len(bare)}")
    print(f"[demo]   with transforms    : {len(has_xform)}")

    # ── Build DebugArchetype — register every node ───────────────────────
    import graph_solver as gs
    from dataclasses import replace

    debug_arch = gs.DebugArchetype()
    for node in nodes:
        debug_arch.register_node(node.key)
    print(f"\n[demo] DebugArchetype registered {len(debug_arch.node_index)} node keys")

    # Override every node's archetype_key so dispatcher routes them through
    # the debug archetype.  Nodes that already had a type-specific archetype
    # from materialize_network will be superseded — this is intentional for
    # the debug test.
    patched_nodes = [replace(n, archetype_key="debug") for n in nodes]

    # ── Inject dummy identity transforms for nodes with no transform ─────
    n_injected = 0
    patched_nodes2 = []
    for node in patched_nodes:
        if node.transform is None and node.analytic_module is None:
            node = replace(node, transform=_identity_transform)
            n_injected += 1
        patched_nodes2.append(node)
    patched_nodes = patched_nodes2
    print(f"[demo]   dummy transforms injected: {n_injected}")

    # ── Build GraphSolver ─────────────────────────────────────────────────
    print("\n[demo] Building GraphSolver …")
    solver = gs.GraphSolver(
        nodes=patched_nodes,
        edges=edges,
        sample_rate=args.sr,
        archetypes={"debug": debug_arch},
    )
    print(f"[demo]   Condensed tier      : {solver.condensed.tier}")
    print(f"[demo]   SCCs                : {len(solver.condensed.sccs)}")
    print(f"[demo]   Registered archetypes: {list(solver.archetypes.keys())}")

    # ── Run ticks ────────────────────────────────────────────────────────
    print(f"\n[demo] Running {args.ticks} solver tick(s) …")
    cdtype = torch.complex128
    ext: dict[str, torch.Tensor] = {}
    for node in patched_nodes:
        if node.layer == "system_in":
            ext[node.key] = torch.tensor(440.0 + 0j, dtype=cdtype)

    all_outputs = []
    for tick in range(args.ticks):
        out = solver.step(ext)
        all_outputs.append(out)
        sample_key = next(
            (n.key for n in patched_nodes if n.layer in ("system_out", "master")),
            next(iter(out), None),
        )
        if sample_key and sample_key in out:
            v = out[sample_key]
            print(f"  tick {tick:02d}  {sample_key!r:40s}  {v.item()!r}")
        else:
            print(f"  tick {tick:02d}  (no output node in result)")

    # ── Summary ───────────────────────────────────────────────────────────
    active_keys = [k for k, v in all_outputs[-1].items()
                   if v is not None and torch.any(v != 0)]
    print(f"\n[demo] Done.  {len(active_keys)}/{len(all_outputs[-1])} nodes "
          f"had non-zero output on final tick.")

    # ── DebugArchetype dispatch report ───────────────────────────────────
    total_dispatches = sum(debug_arch.hit_counts.values())
    print(f"\n[demo] DebugArchetype — total dispatch calls: {total_dispatches}")
    print(f"[demo]   unique keys fired  : {len(debug_arch.hit_counts)}")
    print(f"[demo]   keys registered    : {len(debug_arch.node_index)}")
    never_fired = sorted(
        k for k in debug_arch.node_index if k not in debug_arch.hit_counts
    )
    if never_fired:
        print(f"[demo]   keys never fired  : {len(never_fired)}")
        for k in never_fired[:20]:
            print(f"    {k}")
        if len(never_fired) > 20:
            print(f"    … and {len(never_fired)-20} more")
    else:
        print("[demo]   all registered keys fired at least once ✓")

    # Summarise hit counts per node (sorted by count descending, top 20)
    print(f"\n[demo] Top node dispatch counts (final tick):")
    top = sorted(debug_arch.hit_counts.items(), key=lambda x: -x[1])[:20]
    for k, cnt in top:
        print(f"  {cnt:4d}x  {k}")

    # Report which analytic types were wrapped
    type_counts: dict[str, int] = {}
    for node in patched_nodes:
        if node.analytic_module is not None:
            inner = getattr(node.analytic_module, "analytic_obj", node.analytic_module)
            t = type(inner).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        print("\n[demo] Analytic types wrapped:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {c:4d}  {t}")


if __name__ == "__main__":
    main()
