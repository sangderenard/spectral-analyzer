"""One-shot scrub: strip albedo_rgb / emission_rgb / transmission_rgb literals
from every material YAML and inject `color_profile_name` (and emit/remit names
where applicable).  Run-once script; not part of the runtime.
"""
from __future__ import annotations

import os
import re
import sys

MATDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "configs", "materials")

# material file basename → library color key
COLOR_MAP = {
    "acrylic_pmma":          "acrylic",
    "basic_led_display":     "led_screen_face",
    "basic_paneling":        "basic_paneling",
    "binder_resin":          "binder_resin",
    "bone_nut":              "bone_nut",
    "borosilicate_glass":    "borosilicate",
    "concrete_wall":         "concrete",
    "control_chip":          "control_chip",
    "copper_trace":          "copper",
    "crystal_water_glass":   "crystal_water_glass",
    "emitter_dust":          "emitter_dust",
    "fiber_mesh":            "fiber_mesh",
    "flint_glass":           "flint_glass",
    "interior_cavity":       "interior_cavity",
    "mahogany_body":         "mahogany",
    "nitrocellulose_lacquer":"nitrocellulose_lacquer",
    "polymer_film":          "polymer_film",
    "raw_regolith":          "raw_regolith",
    "rosewood_fretboard":    "rosewood",
    "silica_sand":           "silica_sand",
    "soda_lime_glass":       "soda_lime_glass",
    "spruce_top":            "spruce",
    "stage_floor":           "stage_floor",
    "steel_strings":         "steel",
    "surface_coat":          "surface_coat",
}

# Keys to strip outright (with their inline comments)
STRIP_KEYS = ("albedo_rgb", "emission_rgb", "transmission_rgb")


def scrub_one(path: str, color_key: str | None) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    out: list[str] = []
    changed = False
    inserted_color = False
    for ln in lines:
        stripped = ln.lstrip()
        # Identify the offending top-level keys
        if any(stripped.startswith(k + ":") or stripped.startswith(k + " ")
               for k in STRIP_KEYS):
            changed = True
            # Replace the first occurrence of an albedo line with the
            # color_profile_name entry (preserving leading indentation if any).
            if (not inserted_color and color_key is not None
                    and stripped.startswith("albedo_rgb")):
                indent = ln[:len(ln) - len(stripped)]
                out.append(f"{indent}color_profile_name: {color_key}\n")
                inserted_color = True
            continue
        out.append(ln)
    # Safety: if file had no albedo line at all but we still want a color
    # binding, append it at the end.
    if color_key is not None and not inserted_color:
        out.append(f"color_profile_name: {color_key}\n")
        changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    return changed


def main() -> None:
    touched = 0
    for fname in sorted(os.listdir(MATDIR)):
        if not fname.endswith(".yaml"):
            continue
        base = fname[:-5]
        if base.startswith("test_"):
            continue   # handled separately (rename pass)
        ck = COLOR_MAP.get(base)
        if ck is None:
            print(f"[skip] no color mapping for {base}", file=sys.stderr)
            continue
        if scrub_one(os.path.join(MATDIR, fname), ck):
            print(f"[scrub] {fname} -> color_profile_name: {ck}")
            touched += 1
    print(f"\nDone. {touched} files updated.")


if __name__ == "__main__":
    main()
