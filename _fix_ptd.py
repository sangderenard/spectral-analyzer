"""Remove PerformerConfig block and fix return in patch_to_driver.py."""
with open("patch_to_driver.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Find line indices (0-based)
start_idx = None
return_idx = None
empty_configs_def = None
empty_configs_end = None

for i, l in enumerate(lines):
    if "# \u2500\u2500 PerformerConfig" in l and start_idx is None:
        start_idx = i
    if "return cfg, pcfg, performers, driver_list" in l and return_idx is None:
        return_idx = i
    if "def _empty_configs(" in l and empty_configs_def is None:
        empty_configs_def = i
    if empty_configs_def is not None and i > empty_configs_def and l.strip() == "return cfg, pcfg":
        empty_configs_end = i

print(f"PerformerConfig block start: {start_idx+1 if start_idx else None}")
print(f"return cfg,pcfg line: {return_idx+1 if return_idx else None}")
print(f"_empty_configs def: {empty_configs_def+1 if empty_configs_def else None}")
print(f"_empty_configs end: {empty_configs_end+1 if empty_configs_end else None}")

if start_idx is None or return_idx is None:
    print("ERROR: could not find markers")
    exit(1)

# Replace lines[start_idx..return_idx] (inclusive) with single return line
new_lines = lines[:start_idx] + ["    return cfg, performers, driver_list\n"] + lines[return_idx+1:]

print(f"After first fix: {len(new_lines)} lines")

# Now fix _empty_configs -> _empty_config
# Find them again in new_lines
ec_def = None
ec_return = None
for i, l in enumerate(new_lines):
    if "def _empty_configs(" in l:
        ec_def = i
    if ec_def is not None and i > ec_def and "return cfg, pcfg" in l:
        ec_return = i
        break

print(f"_empty_configs def: {ec_def+1 if ec_def else None}, return: {ec_return+1 if ec_return else None}")

if ec_def is not None and ec_return is not None:
    # Replace the entire _empty_configs function
    new_func = [
        "def _empty_config(V: int, device: torch.device) -> DriverConfig:\n",
        '    """Return a zero-slot DriverConfig when there are no active driver slots."""\n',
        "    def _ft():   return torch.zeros(0, dtype=torch.float64, device=device)\n",
        "    def _it():   return torch.zeros(0, dtype=torch.int64,   device=device)\n",
        "    def _bt():   return torch.zeros(0, dtype=torch.bool,    device=device)\n",
        "    def _ft2(h): return torch.zeros(0, h, dtype=torch.float64, device=device)\n",
        "\n",
        "    return DriverConfig(\n",
        "        D=0, H=1, K=5, V=V, device=device,\n",
        "        f0=_ft(), amplitude=_ft(), phase_origin=_ft(),\n",
        "        pre_delay_samples=_it(), note_duration=_ft(), active=_bt(),\n",
        "        chirp_type=_it(), chirp_f_start=_ft(), chirp_f_end=_ft(),\n",
        "        chirp_tau=_ft(), chirp_power=_ft(),\n",
        "        h_ratios=_ft2(1), h_amps=_ft2(1), n_harmonics=_it(),\n",
        "        env_t=_ft2(5), env_v=_ft2(5), env_n=_it(),\n",
        "        voice_idx=_it(), instrument_idx=_it(),\n",
        "        fm_source_voice=_it(), fm_depth_hz=_ft(),\n",
        "        am_source_voice=_it(), am_depth=_ft(),\n",
        "    )\n",
    ]
    new_lines = new_lines[:ec_def] + new_func + new_lines[ec_return+1:]
    print(f"After _empty_configs fix: {len(new_lines)} lines")

# Also fix the separator comment line for _empty_configs
for i, l in enumerate(new_lines):
    if "Empty-config helper" in l:
        print(f"  separator at line {i+1}: {repr(l)}")

with open("patch_to_driver.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Written.")
