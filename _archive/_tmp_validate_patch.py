"""Validate the orchestral_placement_test patch loads and exercises placement logic."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

# 1 ── Load the JSON
path = os.path.join("presets", "patch", "orchestral_placement_test.json")
raw = json.load(open(path, encoding="utf-8"))
print(f"Loaded {path}: {len(raw['voices'])} voices")

# 2 ── Import and reconstruct
from analytic_driver import AnalyticPatch, resolve_parts_from_patch
patch = AnalyticPatch.from_dict(raw)
print(f"Deserialized: {len(patch.voices)} voices, placement_resonator.enabled={patch.placement_resonator.enabled}")
print(f"  layout_mode={patch.placement_resonator.layout_mode}")
print(f"  room_radius={patch.placement_resonator.room_radius}  room_height={patch.placement_resonator.room_height}")
print(f"  receiver_array_key={patch.placement_resonator.receiver_array_key}")

# 2b ── Trigger part resolution (normally happens at play time)
#       We need _arrangement_metrics for resolve_parts_from_patch, so
#       just call _refresh_part_placement_layout on a minimal parts list.
from analytic_driver import _refresh_part_placement_layout, Part
# Build minimal parts from voices grouped by (register, seq_role, voice_role)
from collections import defaultdict
groups = defaultdict(list)
for v in patch.voices:
    gk = f"{v.register}-{v.seq_role}-{v.voice_role}"
    groups[gk].append(v.key)
for gk, vkeys in groups.items():
    reg, sr, vr = gk.split("-")
    patch.parts.append(Part(
        key=gk, label=gk, register=reg, seq_role=sr, voice_role=vr,
        voice_keys=vkeys, player_count=1,
    ))
_refresh_part_placement_layout(patch)

# 3 ── Check parts were created by _refresh_part_placement_layout
print(f"\nParts ({len(patch.parts)}):")
for pt in patch.parts:
    n_chairs = len(pt.chairs)
    n_perf = sum(len(c.performers) for c in pt.chairs)
    print(f"  {pt.key:20s} reg={pt.register:6s} seq={pt.seq_role:8s} vrole={pt.voice_role:10s} "
          f"chairs={n_chairs} performers={n_perf}")
    for ch in pt.chairs:
        for pf in ch.performers:
            print(f"    performer {pf.key}: x={pf.x:+6.2f} y={pf.y:+6.2f} z={pf.z:+5.2f} "
                  f"r={pf.radius:.2f} a={pf.angle_deg:+7.2f}")

# 4 ── Round-trip check
d2 = patch.to_dict()
patch2 = AnalyticPatch.from_dict(d2)
assert len(patch2.parts) == len(patch.parts), "Round-trip lost parts!"
for p1, p2 in zip(patch.parts, patch2.parts):
    assert p1.key == p2.key
    for c1, c2 in zip(p1.chairs, p2.chairs):
        for pf1, pf2 in zip(c1.performers, c2.performers):
            assert abs(pf1.z - pf2.z) < 1e-9, f"z lost in round-trip for {pf1.key}"
print("\nRound-trip OK — all fields including z survive save/load")

# 5 ── Check placement_resonator survives
pr2 = patch2.placement_resonator
assert pr2.enabled == True
assert pr2.layout_mode == "stage"
assert pr2.room_radius == 12.0
assert pr2.receiver_array_key == "binaural_standard"
print("PlacementResonatorConfig round-trip OK")
