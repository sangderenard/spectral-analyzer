#version 430

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer HitBuf {
    float hit[];
};

layout(std430, binding = 1) coherent buffer FieldAccum {
    int acc[];
};

uniform int n_hits;
uniform int hit_stride;
uniform int n_bands;
uniform int nx;
uniform int ny;
uniform int nz;
uniform vec3 bmin;
uniform vec3 bmax;
uniform float fixed_scale;

const int MAX_BANDS = 32;

int clamp_idx(float v, float vmin, float dv, int n) {
    int i = int(floor((v - vmin) / dv));
    return clamp(i, 0, n - 1);
}

void add_voxel(int x, int y, int z, float w, int row) {
    if (w <= 0.0) return;
    int cell = (z * ny + y) * nx + x;
    int n_cells = nx * ny * nz;
    int bands = min(n_bands, MAX_BANDS);
    for (int b = 0; b < bands; ++b) {
        float re = hit[row + 28 + b] * w;
        float im = hit[row + 28 + MAX_BANDS + b] * w;
        int ire = int(clamp(round(re * fixed_scale), -2147480000.0, 2147480000.0));
        int iim = int(clamp(round(im * fixed_scale), -2147480000.0, 2147480000.0));
        int base = 2 * (b * n_cells + cell);
        atomicAdd(acc[base + 0], ire);
        atomicAdd(acc[base + 1], iim);
    }
}

void add_trilinear(vec3 p, int row) {
    vec3 span = max(bmax - bmin, vec3(1.0e-9));
    float ux = (p.x - bmin.x) / span.x * float(nx - 1);
    float uy = (p.y - bmin.y) / span.y * float(ny - 1);
    float uz = (p.z - bmin.z) / span.z * float(nz - 1);
    if (ux < 0.0 || uy < 0.0 || uz < 0.0 || ux >= float(nx) || uy >= float(ny) || uz >= float(nz))
        return;

    int x0 = int(ux), y0 = int(uy), z0 = int(uz);
    int x1 = min(x0 + 1, nx - 1);
    int y1 = min(y0 + 1, ny - 1);
    int z1 = min(z0 + 1, nz - 1);
    float fx = ux - float(x0);
    float fy = uy - float(y0);
    float fz = uz - float(z0);

    add_voxel(x0, y0, z0, (1.0-fx)*(1.0-fy)*(1.0-fz), row);
    add_voxel(x1, y0, z0,      fx *(1.0-fy)*(1.0-fz), row);
    add_voxel(x0, y1, z0, (1.0-fx)*     fy *(1.0-fz), row);
    add_voxel(x1, y1, z0,      fx *     fy *(1.0-fz), row);
    add_voxel(x0, y0, z1, (1.0-fx)*(1.0-fy)*     fz,  row);
    add_voxel(x1, y0, z1,      fx *(1.0-fy)*     fz,  row);
    add_voxel(x0, y1, z1, (1.0-fx)*     fy *     fz,  row);
    add_voxel(x1, y1, z1,      fx *     fy *     fz,  row);
}

void main() {
    uint hi = gl_GlobalInvocationID.x;
    if (hi >= uint(n_hits)) return;
    int row = int(hi) * hit_stride;

    vec3 p0 = vec3(hit[row + 9], hit[row + 10], hit[row + 11]);
    vec3 p1 = vec3(hit[row + 0], hit[row + 1], hit[row + 2]);
    vec3 d = p1 - p0;
    vec3 cell = (bmax - bmin) / vec3(max(nx, 1), max(ny, 1), max(nz, 1));
    if (length(d) <= 1.0e-12) {
        add_trilinear(p1, row);
        return;
    }

    int ix = clamp_idx(p0.x, bmin.x, cell.x, nx);
    int iy = clamp_idx(p0.y, bmin.y, cell.y, ny);
    int iz = clamp_idx(p0.z, bmin.z, cell.z, nz);
    int stepx = (d.x > 0.0) ? 1 : ((d.x < 0.0) ? -1 : 0);
    int stepy = (d.y > 0.0) ? 1 : ((d.y < 0.0) ? -1 : 0);
    int stepz = (d.z > 0.0) ? 1 : ((d.z < 0.0) ? -1 : 0);

    float inf = 1.0e30;
    float dtx = (abs(d.x) > 1.0e-15) ? abs(cell.x / d.x) : inf;
    float dty = (abs(d.y) > 1.0e-15) ? abs(cell.y / d.y) : inf;
    float dtz = (abs(d.z) > 1.0e-15) ? abs(cell.z / d.z) : inf;

    float tx = (stepx == 0) ? inf : ((bmin.x + float(stepx > 0 ? ix + 1 : ix) * cell.x) - p0.x) / d.x;
    float ty = (stepy == 0) ? inf : ((bmin.y + float(stepy > 0 ? iy + 1 : iy) * cell.y) - p0.y) / d.y;
    float tz = (stepz == 0) ? inf : ((bmin.z + float(stepz > 0 ? iz + 1 : iz) * cell.z) - p0.z) / d.z;

    float t_prev = 0.0;
    int guard_max = nx + ny + nz + 1024;
    for (int guard = 0; guard < guard_max && t_prev <= 1.0; ++guard) {
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) break;
        float t_next = min(tx, min(ty, tz));
        if (t_next > 1.0e20) t_next = 1.0;
        t_next = max(t_prev, min(1.0, t_next));
        vec3 pm = p0 + d * (0.5 * (t_prev + t_next));
        add_trilinear(pm, row);
        if (t_next >= 1.0) break;
        if (tx <= ty && tx <= tz) {
            ix += stepx; tx += dtx;
        } else if (ty <= tx && ty <= tz) {
            iy += stepy; ty += dty;
        } else {
            iz += stepz; tz += dtz;
        }
        t_prev = t_next;
    }
}
