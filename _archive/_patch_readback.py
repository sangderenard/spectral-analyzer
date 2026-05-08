"""
Patch all glGetBufferSubData(..., arr) calls to use
ctypes.c_void_p(arr.ctypes.data) so PyOpenGL writes into the numpy array.

Also ensures `import ctypes` is present in each affected file.
"""
import re, pathlib

PATTERN = re.compile(
    r'glGetBufferSubData\(GL_SHADER_STORAGE_BUFFER,\s*0,\s*(\w+)\.nbytes,\s*\1\)'
)

def patch_file(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")

    # Check if ctypes is already imported
    needs_ctypes = bool(PATTERN.search(src))
    if not needs_ctypes:
        print(f"{path.name}: no matches, skipping")
        return

    # Replace all matching calls
    def replacer(m):
        var = m.group(1)
        return (
            f"glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, {var}.nbytes, "
            f"ctypes.c_void_p({var}.ctypes.data))"
        )

    new_src, n = PATTERN.subn(replacer, src)
    print(f"{path.name}: {n} replacements")

    # Add `import ctypes` after the first `import ` line if not already present
    if "import ctypes" not in new_src:
        new_src = "import ctypes\n" + new_src
        print(f"  -> added `import ctypes` at top")

    path.write_text(new_src, encoding="utf-8")


ROOT = pathlib.Path(__file__).parent
for fname in ("acoustic_amr.py", "test_amr_throughput.py"):
    patch_file(ROOT / fname)

print("Done.")
