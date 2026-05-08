import os

with open('demo_pluck_gl.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Find the exact bounds of _GPU_RAY_FIELD_CS = """..."""
marker = '_GPU_RAY_FIELD_CS = """\n'
start = src.index(marker) + len(marker)
end = src.index('\n"""', start) + 1  # points to the \n before the closing """

glsl = src[start:end]

out_path = os.path.join('csrc', 'shaders', 'ray_tracer.comp.glsl')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(glsl)

print(f'Wrote {len(glsl.splitlines())} lines to {out_path}')
