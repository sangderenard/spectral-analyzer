"""Quick smoke test: verify all body scenes build and section coupling runs."""
import random
from sm_plugins.orchestral_resonance import (
    _build_body_scene, _BODY_PANEL_BUILDERS, _BODY_PROFILES,
)

rng = random.Random(42)
for bt in _BODY_PANEL_BUILDERS:
    s = _build_body_scene(bt, jitter_rng=rng)
    if s is None:
        print(f"{bt:15s}: None (pass-through)")
    else:
        n_baffles = len(s.geometry.baffles)
        n_src = len(s.sources)
        n_recv = len(s.receivers)
        n_ap = len(s.apertures)
        print(f"{bt:15s}: {n_baffles} baffles, {n_src} src, {n_recv} recv, {n_ap} ap")
        assert n_src == 1
        assert n_recv == 1
        assert n_ap == 1
        assert n_baffles > 0

print("\nAll body scenes OK")

# Quick check: PARAM_SPECS has the new coupling params
from sm_plugins.orchestral_resonance import PARAM_SPECS
param_names = [p["name"] for p in PARAM_SPECS]
assert "section_coupling_gain" in param_names, "Missing section_coupling_gain"
assert "section_reaction_lag_ms" in param_names, "Missing section_reaction_lag_ms"
print("Section coupling params present in PARAM_SPECS")
