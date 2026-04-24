"""Audit patch structure — read only."""
import json

d = json.load(open("analytic_patch.json"))

print("=== Patch structure ===")
print(f"voices: {len(d.get('voices', []))}")
print(f"lfos: {len(d.get('lfos', []))}")
print(f"modules: {len(d.get('modules', []))}")
print(f"controls: {len(d.get('controls', []))}")
print(f"mixers: {len(d.get('mixers', []))}")
print(f"param_nodes: {len(d.get('param_nodes', []))}")
print(f"parts: {len(d.get('parts', []))}")

r = d.get("routing", {})
edges = r.get("edges", [])
fb = r.get("feedback", {})
print(f"routing edges: {len(edges)}")
print(f"routing feedback enabled: {fb.get('enabled')}")
print(f"routing feedback iterations: {fb.get('iterations')}")
print(f"routing feedback decay: {fb.get('decay')}")

node_set = set()
for e in edges:
    node_set.add(e.get("source", ""))
    node_set.add(e.get("target", ""))
print(f"routing unique nodes: {len(node_set)}")

print("\n=== Voices ===")
for i, v in enumerate(d.get("voices", [])):
    chirp = v.get("chirp", {})
    emission = v.get("emission_mode", "single")
    gran = v.get("granular")
    body = v.get("body_type", "direct")
    poly = v.get("polyphony_mode", "")
    harms = len(v.get("harmonics", []))
    env_segs = len(v.get("envelope", {}).get("segments", []))
    muted = v.get("muted", False)
    print(f"  [{i}] key={v.get('key','')} emission={emission} body={body} "
          f"poly={poly} harmonics={harms} env_segs={env_segs} "
          f"chirp={chirp.get('chirp_type','')} muted={muted}")

print("\n=== Modules ===")
for m in d.get("modules", []):
    print(f"  type={m.get('module_type')} plugin={m.get('sm_plugin')} "
          f"key={m.get('key')} muted={m.get('muted')}")

print("\n=== Parts ===")
for pt in d.get("parts", []):
    chairs = pt.get("chairs", [])
    n_perf = sum(len(c.get("performers", [])) for c in chairs)
    print(f"  part={pt.get('key','')} chairs={len(chairs)} performers={n_perf} "
          f"voice_keys={pt.get('voice_keys',[])}")

print("\n=== Edges (first 30) ===")
for e in edges[:30]:
    print(f"  {e.get('source','')} -> {e.get('target','')}  gain={e.get('gain',1.0)}")
if len(edges) > 30:
    print(f"  ... ({len(edges)} total)")

print("\n=== Placement resonator ===")
pr = d.get("placement_resonator")
print(f"  present in JSON: {pr is not None}")
if pr:
    for k, v in pr.items():
        print(f"  {k}: {v}")
