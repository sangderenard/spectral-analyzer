#version 430 core
/* Scatter BDPT side records into already-packed T5 rows.
 *
 * bdpt_pack_t5 is kept O(vertices).  This pass is O(side-records * log(vertices)):
 * each spectral/PDF/optical record binary-searches the sorted vertex key and
 * writes directly into the matching light/camera T5 row.
 */
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer SortKeysBuf { uint  sort_keys[];  };
layout(std430, binding = 1) readonly buffer BdptBuf     { float bdpt_verts[]; };
layout(std430, binding = 2)          buffer T5LightBuf  { float t5_light[];   };
layout(std430, binding = 3)          buffer T5CamBuf    { float t5_cam[];     };

#define BDPT_SPECTRAL_STRIDE 8
#define BDPT_PDF_STRIDE      12
#define BDPT_OPTICAL_STRIDE  28
#define T5_LGV_STRIDE        56
#define T5_CGV_STRIDE        72
#define LGV_BAND_BASE        16
#define CGV_BAND_BASE        22
#define LGV_EDGE_FWD         54
#define LGV_EDGE_BWD         55
#define CGV_EDGE_FWD         60
#define CGV_EDGE_BWD         61
#define BDPT_OPT_TIR         2u
#define BDPT_OPT_APERTURE_CLIP 3u
#define BDPT_OPT_VIGNETTE_CLIP 4u
#define BDPT_OPT_ABSORPTION  5u

uniform int scatter_mode;       /* 0=spectral, 1=optical, 2=pdf */
uniform int nv;
uniform int n_lv;
uniform int n_records;
uniform int record_offset;
uniform int n_bands;
uniform int bdpt_spectral_base;
uniform int bdpt_pdf_base;
uniform int bdpt_optical_base;

int find_sorted_pos(uint stream, uint sid, uint vi) {
    uint key_hi = (stream << 31) | (sid & 0x7fffffffu);
    uint key_lo = vi << 16;
    int lo = 0;
    int hi = nv;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint mh = sort_keys[mid * 2 + 0];
        uint ml = sort_keys[mid * 2 + 1];
        bool less = (mh < key_hi) || (mh == key_hi && ml < key_lo);
        if (less) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= nv) return -1;
    uint fh = sort_keys[lo * 2 + 0];
    uint fl = sort_keys[lo * 2 + 1];
    return (fh == key_hi && fl == key_lo) ? lo : -1;
}

void scatter_spectral(uint r) {
    int rb = bdpt_spectral_base + int(r) * BDPT_SPECTRAL_STRIDE;
    uint sid = floatBitsToUint(bdpt_verts[rb + 0]);
    uint vi_band = floatBitsToUint(bdpt_verts[rb + 1]);
    uint vi = (vi_band >> 16) & 0xffffu;
    uint band = vi_band & 0xffffu;
    if (band >= uint(min(n_bands, 32))) return;
    float re = bdpt_verts[rb + 2];
    float im = bdpt_verts[rb + 3];
    float beta = sqrt(max(0.0, re * re + im * im));

    int p = find_sorted_pos(0u, sid, vi);
    if (p >= 0 && p < n_lv) {
        int ob = p * T5_LGV_STRIDE;
        t5_light[ob + LGV_BAND_BASE + int(band)] = beta;
        return;
    }
    p = find_sorted_pos(1u, sid, vi);
    if (p >= n_lv) {
        int ob = (p - n_lv) * T5_CGV_STRIDE;
        t5_cam[ob + CGV_BAND_BASE + int(band)] = beta;
    }
}

float pdf_area_to_target(float pf, float pr, float pa, float ps, bool reverse_pdf,
                         vec3 sampler_pos, vec3 target_pos, vec3 target_nrm) {
    if (!reverse_pdf && pa > 0.0 && !isnan(pa) && !isinf(pa))
        return pa;

    vec3 d = target_pos - sampler_pos;
    float dist2 = dot(d, d);
    if (dist2 <= 1e-18)
        return 1e-12;
    d *= inversesqrt(dist2);

    float cos_to = max(0.0, abs(dot(target_nrm, -d)));
    if (cos_to <= 0.0)
        return 0.0;

    float p = reverse_pdf ? pr : ((ps > 0.0 && !isnan(ps) && !isinf(ps)) ? ps : pf);
    if (p <= 0.0 || isnan(p) || isinf(p))
        return 0.0;
    return max(p * cos_to / dist2, 1e-12);
}

void scatter_pdf_to_stream(uint stream, uint sid, uint vi,
                           float pf, float pr, float pa, float ps, uint flags) {
    int p = find_sorted_pos(stream, sid, vi);
    if (p < 0 || p + 1 >= nv) return;
    uint key_hi = sort_keys[p * 2 + 0];
    if (sort_keys[(p + 1) * 2 + 0] != key_hi) return;

    if (stream == 0u) {
        if (p >= n_lv || p + 1 >= n_lv) return;
        int ob = p * T5_LGV_STRIDE;
        int nb = (p + 1) * T5_LGV_STRIDE;
        vec3 pos = vec3(t5_light[ob + 0], t5_light[ob + 1], t5_light[ob + 2]);
        vec3 nrm = vec3(t5_light[ob + 3], t5_light[ob + 4], t5_light[ob + 5]);
        vec3 nxt_pos = vec3(t5_light[nb + 0], t5_light[nb + 1], t5_light[nb + 2]);
        vec3 nxt_nrm = vec3(t5_light[nb + 3], t5_light[nb + 4], t5_light[nb + 5]);
        t5_light[ob + 11] = pf;
        t5_light[ob + 12] = pr;
        t5_light[ob + 13] = uintBitsToFloat(flags);
        t5_light[ob + LGV_EDGE_FWD] = pdf_area_to_target(pf, pr, pa, ps, false,
                                                          pos, nxt_pos, nxt_nrm);
        t5_light[ob + LGV_EDGE_BWD] = pdf_area_to_target(pf, pr, pa, ps, true,
                                                          nxt_pos, pos, nrm);
    } else {
        if (p < n_lv) return;
        int ob = (p - n_lv) * T5_CGV_STRIDE;
        int nb = (p + 1 - n_lv) * T5_CGV_STRIDE;
        vec3 pos = vec3(t5_cam[ob + 0], t5_cam[ob + 1], t5_cam[ob + 2]);
        vec3 nrm = vec3(t5_cam[ob + 3], t5_cam[ob + 4], t5_cam[ob + 5]);
        vec3 nxt_pos = vec3(t5_cam[nb + 0], t5_cam[nb + 1], t5_cam[nb + 2]);
        vec3 nxt_nrm = vec3(t5_cam[nb + 3], t5_cam[nb + 4], t5_cam[nb + 5]);
        t5_cam[ob + 15] = pf;
        t5_cam[ob + 16] = pr;
        t5_cam[ob + 17] = uintBitsToFloat(flags);
        t5_cam[ob + CGV_EDGE_FWD] = pdf_area_to_target(pf, pr, pa, ps, false,
                                                        pos, nxt_pos, nxt_nrm);
        t5_cam[ob + CGV_EDGE_BWD] = pdf_area_to_target(pf, pr, pa, ps, true,
                                                        nxt_pos, pos, nrm);
    }
}

void scatter_pdf(uint r) {
    int rb = bdpt_pdf_base + int(r) * BDPT_PDF_STRIDE;
    uint sid = floatBitsToUint(bdpt_verts[rb + 0]);
    uint packed_vm = floatBitsToUint(bdpt_verts[rb + 1]);
    uint vi = packed_vm & 0xffffu;
    float pf = bdpt_verts[rb + 2];
    float pr = bdpt_verts[rb + 3];
    if (pf < 0.0 || pr < 0.0 || isnan(pf) || isnan(pr) || isinf(pf) || isinf(pr))
        return;
    float pa = bdpt_verts[rb + 4];
    float ps = bdpt_verts[rb + 5];
    uint flags = floatBitsToUint(bdpt_verts[rb + 8]);

    scatter_pdf_to_stream(0u, sid, vi, pf, pr, pa, ps, flags);
    scatter_pdf_to_stream(1u, sid, vi, pf, pr, pa, ps, flags);
}

void scatter_optical(uint r) {
    int rb = bdpt_optical_base + int(r) * BDPT_OPTICAL_STRIDE;
    uint sid = floatBitsToUint(bdpt_verts[rb + 0]);
    uint packed_ve = floatBitsToUint(bdpt_verts[rb + 1]);
    uint vi = packed_ve & 0xffffu;
    uint packed_rf = floatBitsToUint(bdpt_verts[rb + 2]);
    uint reason = packed_rf & 0xffu;
    uint stream = (packed_rf >> 8) & 0xffu;
    int p = find_sorted_pos(stream & 1u, sid, vi);
    if (p < 0) return;

    bool blocked = reason == BDPT_OPT_ABSORPTION ||
                   reason == BDPT_OPT_TIR ||
                   reason == BDPT_OPT_APERTURE_CLIP ||
                   reason == BDPT_OPT_VIGNETTE_CLIP;
    float jac = bdpt_verts[rb + 27];
    if (!(jac > 0.0) || isnan(jac) || isinf(jac))
        jac = 1.0;

    if (stream == 0u && p < n_lv) {
        int ob = p * T5_LGV_STRIDE;
        if (blocked) t5_light[ob + 14] = uintBitsToFloat(1u);
        t5_light[ob + 53] = jac;
        t5_light[ob + 54] *= jac;
        t5_light[ob + 55] *= (1.0 / jac);
    } else if (stream == 1u && p >= n_lv) {
        int ob = (p - n_lv) * T5_CGV_STRIDE;
        if (blocked) t5_cam[ob + 18] = uintBitsToFloat(1u);
        t5_cam[ob + 59] = jac;
        t5_cam[ob + 60] *= jac;
        t5_cam[ob + 61] *= (1.0 / jac);
    }
}

void main() {
    uint r = uint(record_offset) + gl_GlobalInvocationID.x;
    if (int(r) >= n_records) return;
    if (scatter_mode == 0)
        scatter_spectral(r);
    else if (scatter_mode == 1)
        scatter_optical(r);
    else if (scatter_mode == 2)
        scatter_pdf(r);
}
