#version 430

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer FieldAccum {
    int acc[];
};

layout(rgba32f, binding = 0) writeonly uniform image3D field_tex;

uniform int n_cells;
uniform int n_bands;
uniform int nx;
uniform int ny;
uniform int nz;
uniform float inv_fixed_scale;
uniform vec3 rgb_w[32];

void main() {
    uint cell_u = gl_GlobalInvocationID.x;
    if (cell_u >= uint(n_cells)) return;
    int cell = int(cell_u);
    int bands = min(n_bands, 32);

    vec3 rgb = vec3(0.0);
    for (int b = 0; b < bands; ++b) {
        int base = 2 * (b * n_cells + cell);
        float re = float(acc[base + 0]) * inv_fixed_scale;
        float im = float(acc[base + 1]) * inv_fixed_scale;
        rgb += sqrt(max(0.0, re * re + im * im)) * rgb_w[b];
    }

    int x = cell % nx;
    int y = (cell / nx) % ny;
    int z = cell / (nx * ny);
    imageStore(field_tex, ivec3(x, y, z), vec4(max(rgb, vec3(0.0)), 1.0));
}
