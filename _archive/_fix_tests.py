"""Patch tests/test_patch_to_driver.py to remove pcfg references."""
import re

with open("tests/test_patch_to_driver.py", "r", encoding="utf-8") as f:
    src = f.read()

# 1. _build helper: return just cfg
src = src.replace(
    "    cfg, pcfg, _performers, _driver_list = build_driver_config(patch, performers, _DEV, _SR, note_hz=note_hz)\n"
    "    return cfg, pcfg",
    "    cfg, _performers, _driver_list = build_driver_config(patch, performers, _DEV, _SR, note_hz=note_hz)\n"
    "    return cfg"
)

# 2. All unpack patterns at call sites
src = src.replace("cfg, pcfg = _build(", "cfg = _build(")
src = src.replace("cfg, _ = _build(", "cfg = _build(")
src = src.replace("_, pcfg = _build(", "_ = _build(")

# 3. Remove pcfg.P assertions
src = re.sub(r"        assert pcfg\.P == \d+\n", "", src)

# 4. test_tensor_shapes_consistent: fix D,H,K,V,P line and pcfg assertions
src = src.replace(
    "        D, H, K, V, P = cfg.D, cfg.H, cfg.K, cfg.V, pcfg.P",
    "        D, H, K, V = cfg.D, cfg.H, cfg.K, cfg.V"
)
src = src.replace(
    "        assert cfg.performer_idx.shape  == (D,)\n"
    "        assert pcfg.gain.shape          == (P,)\n"
    "        assert pcfg.clan_id.shape       == (P,)\n",
    "        assert cfg.instrument_idx.shape == (D,)\n"
)

# 5. test_performer_gain_not_in_amplitude: remove pcfg.gain assertion
src = src.replace(
    "        # gain ≈ 10 (20 dB)\n"
    "        assert pcfg.gain[0].item() == pytest.approx(10.0, rel=1e-6)\n",
    ""
)

# 6. test_performer_idx_maps_correctly: fix to use instrument_idx
src = src.replace(
    "        pidxs = sorted(cfg.performer_idx.tolist())\n"
    "        assert pidxs == [0, 1]",
    "        pidxs = sorted(cfg.instrument_idx.tolist())\n"
    "        assert pidxs == [0, 1]"
)

# 7. Remove test_clan_ids_grouped_by_chair entirely
clan_test = '''    def test_clan_ids_grouped_by_chair(self):
        voices = [_Voice(key="v0"), _Voice(key="v1"), _Voice(key="v2")]
        perfs  = [
            _Performer(key="p0", source_voice_keys=["v0"], chair_key="violin"),
            _Performer(key="p1", source_voice_keys=["v1"], chair_key="violin"),
            _Performer(key="p2", source_voice_keys=["v2"], chair_key="cello"),
        ]
        patch = _Patch(voices=voices)
        _, pcfg = _build(patch, perfs)
        # Performers in the same chair_key share the same clan_id
        assert pcfg.clan_id[0].item() == pcfg.clan_id[1].item()
        assert pcfg.clan_id[0].item() != pcfg.clan_id[2].item()

'''
src = src.replace(clan_test, "")

# 8. Remove test_performer_gain_not_in_amplitude if it's now empty (just cfg assertion left)
# Check if pcfg still in src
remaining = [line for line in src.splitlines() if "pcfg" in line]
if remaining:
    print("WARNING: remaining pcfg references:")
    for l in remaining:
        print(" ", repr(l))
else:
    print("All pcfg references removed.")

with open("tests/test_patch_to_driver.py", "w", encoding="utf-8") as f:
    f.write(src)
print("Written.")
