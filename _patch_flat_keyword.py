"""
Rename the GLSL variable 'flat' (a reserved keyword) to 'p_idx'
inside shader string constants in acoustic_amr.py.
Python-level 'flat' variables are untouched.
"""
import re

path = "acoustic_amr.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# The GLSL shader constants begin with a triple-quoted string assigned to a
# variable whose name ends in _GLSL.  We iterate over those spans and rename
# 'flat' only within them.

# Match each _GLSL = """..."""  block
pattern = re.compile(r'(_[A-Z0-9_]+_GLSL\s*=\s*""")(.*?)(""")', re.DOTALL)

replacements = 0

def replace_in_glsl(m):
    global replacements
    prefix, body, suffix = m.group(1), m.group(2), m.group(3)
    new_body = body
    # rename variable declaration:  int flat  /  int flat\t
    new_body, n1 = re.subn(r'\bint\s+flat\b', 'int p_idx', new_body)
    # rename all [flat] accesses
    new_body, n2 = re.subn(r'\[flat\]', '[p_idx]', new_body)
    replacements += n1 + n2
    return prefix + new_body + suffix

new_src = pattern.sub(replace_in_glsl, src)

if replacements == 0:
    print("Nothing to patch — 'flat' not found in any GLSL block.")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"Patched {replacements} occurrences of 'flat' → 'p_idx' in GLSL blocks: OK")
