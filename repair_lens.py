"""Repair the broken one-liner sections in lens_manifold.py."""
import pathlib, re

path = pathlib.Path("camera_software/lens_manifold.py")
src = path.read_text(encoding="utf-8")

# ── 1. Identify the broken region ──────────────────────────────────────────
# The blob starts right after the push-method docstring's last real newline
# (which ends with '        stored).\n') and ends right before the
# line that starts '    # -----...flush' section OR the next properly-indented
# method definition line that happens to be on a real line.
#
# Strategy: find the ONE very-long physical line (> 500 chars) that contains
# literal backslash-n sequences and replace it wholesale.

lines = src.split('\n')
broken_line_idx = None
for i, line in enumerate(lines):
    if r'\"\"\"' in line and len(line) > 200:
        broken_line_idx = i
        print(f"Found broken line at index {i}, length={len(line)}")
        break

if broken_line_idx is None:
    print("No broken line found -- may already be fixed.")
    import sys; sys.exit(0)

blob = lines[broken_line_idx]

# ── 2. Unescape the blob to get the actual intended Python code ────────────
# The blob has:
#   \"\"\"  → """
#   \n    → actual newline
#   \u2014 → — (em dash) [these are fine, leave them]
#   \u2264 → ≤
#   \u00d7 → × (multiplication sign – avoid in identifiers/comments outside strings)
#
# We want to turn the one-liner back into properly indented Python.

import codecs

# Step 1: unescape \" → "  (but NOT inside triple-quote boundaries)
# Step 2: unescape \n → newline
# The blob is the literal content; we can use unicode_escape decode trick carefully.

# Replace literal \n with real newlines first
fixed = blob.replace(r'\n', '\n')
# Replace literal \" with "
fixed = fixed.replace(r'\"', '"')
# Replace \u sequences (other than the ones already decoded)
fixed = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), fixed)
# Replace \u2014 (em dash) and similar that might remain
# (already done by the regex above)

# ── 3. Sanity: the fixed blob should now contain real triple-quotes ────────
assert '"""' in fixed, "No triple-quote found after unescaping -- check logic"
print(f"Fixed section has {fixed.count(chr(10))} lines")

# ── 4. Rebuild the file ────────────────────────────────────────────────────
lines[broken_line_idx] = fixed
new_src = '\n'.join(lines)
path.write_text(new_src, encoding="utf-8")
print("Written OK")

# ── 5. Verify with ast.parse ───────────────────────────────────────────────
import ast
try:
    ast.parse(new_src)
    print("AST parse: PASSED")
except SyntaxError as e:
    print(f"AST parse FAILED: {e}")
