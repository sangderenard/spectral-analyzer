"""Drop test_ prefix from internal name fields and remap profile references
to the central spectral library names."""
from __future__ import annotations
import os, re, sys

MATDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "configs", "materials")

# Per-file overrides for internal name + profile references.
# Maps old YAML value → new YAML value.
SUBS = {
    "chrome_mirror.yaml": [
        ("test_chrome_mirror", "chrome_mirror"),
        ("test_color_chrome",  "chrome"),
    ],
    "emerald_emissive.yaml": [
        ("test_emerald_emissive",  "emerald_emissive"),
        ("test_color_emerald_body","emerald_body"),
        ("test_emerald_internal",  "emerald_internal"),
    ],
    "fibonacci_satin_gold.yaml": [
        ("test_fibonacci_satin_gold", "fibonacci_satin_gold"),
        ("test_color_satin_gold",     "satin_gold"),
    ],
    "hyperbolic_slate.yaml": [
        ("test_hyperbolic_slate",   "hyperbolic_slate"),
        ("test_color_slate_body",   "slate_body"),
    ],
    "obsidian_polish.yaml": [
        ("test_obsidian_polish",     "obsidian_polish"),
        ("test_color_obsidian_body", "obsidian_body"),
    ],
    "ruby_glass.yaml": [
        ("test_ruby_glass",        "ruby_glass"),
        ("test_color_ruby_body",   "ruby_body"),
        ("test_ruby_internal",     "ruby_internal"),
    ],
    "sapphire_emissive.yaml": [
        ("test_sapphire_emissive",   "sapphire_emissive"),
        ("test_color_sapphire_body", "sapphire_body"),
        ("test_sapphire_internal",   "sapphire_internal"),
    ],
}


def main() -> None:
    for fname, subs in SUBS.items():
        path = os.path.join(MATDIR, fname)
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read()
        before = txt
        for old, new in subs:
            txt = txt.replace(old, new)
        if txt != before:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
            print(f"[rewrote] {fname}")
        else:
            print(f"[noop]    {fname}")


if __name__ == "__main__":
    main()
