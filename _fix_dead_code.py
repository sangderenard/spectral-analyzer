import re

path = r"c:\Users\alber\Downloads\spectral-analyzer\parametric_curve_editor.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

# New evaluate_normalized body ends with this line (8-space indent)
new_end_marker   = "        return z.reshape(orig_shape)\n"
# Old dead _ensure_baked tail ends with:
old_ensure_end   = "        return self._seg_poles, self._seg_residues, self._seg_breaks\n"

# Step 1: remove dead Stage-2/3 tail that falls between the two markers
idx_new_end = src.find(new_end_marker)
assert idx_new_end != -1, "Could not find new evaluate_normalized end"

idx_old_ensure_end = src.find(old_ensure_end, idx_new_end)
assert idx_old_ensure_end != -1, "Could not find old _ensure_baked tail"

delete_from = idx_new_end + len(new_end_marker)
delete_to   = idx_old_ensure_end + len(old_ensure_end)
src = src[:delete_from] + src[delete_to:]
print("Step 1 done: removed dead Stage-2/3 tail")

# Step 2: remove the old "# ── evaluation" section + old evaluate_normalized body
old_eval_section_start = "\n    # \u2500\u2500 evaluation \u2500"
idx_eval_section = src.find(old_eval_section_start, idx_new_end)
assert idx_eval_section != -1, f"Could not find old evaluation section, searched from {idx_new_end}"

idx_bake = src.find("\n    def bake(", idx_eval_section)
assert idx_bake != -1, "Could not find bake() after old eval section"

src = src[:idx_eval_section] + src[idx_bake:]
print("Step 2 done: removed old evaluate_normalized and evaluation header")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("File written successfully")
