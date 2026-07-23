/* Canonical GLSL Jones/differential operators. Declares no SSBO binding. */
#ifndef COMPLEX_OPTICAL_OPERATORS_GLSL_INC
#define COMPLEX_OPTICAL_OPERATORS_GLSL_INC

#define COMPLEX_OPTICAL_OPERATOR_SCHEMA_VERSION 1u
#define COMPLEX_BASIS_WORDS 8u
#define COMPLEX_OPERATOR_WORDS 24u
#define COMPLEX_SOURCE_MODE_WORDS 12u

struct ComplexTransverseBasis {
    vec4 s;
    vec4 p;
};

struct ComplexOpticalOperator {
    vec4 jones_row0;
    vec4 jones_row1;
    mat4 phase_space;
};

struct ComplexSourceMode {
    vec4 amplitudes;
    float power_weight;
    uint coherence_lo;
    uint coherence_hi;
    uint basis_id;
    uint operator_id;
    uint flags;
    uint reserved0;
    uint reserved1;
};

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

void complex_jones_apply(
    ComplexOpticalOperator op,
    vec2 in_s,
    vec2 in_p,
    out vec2 out_s,
    out vec2 out_p
) {
    out_s = complex_mul(op.jones_row0.xy, in_s)
          + complex_mul(op.jones_row0.zw, in_p);
    out_p = complex_mul(op.jones_row1.xy, in_s)
          + complex_mul(op.jones_row1.zw, in_p);
}

mat2 complex_basis_change(
    ComplexTransverseBasis source,
    ComplexTransverseBasis destination
) {
    return mat2(
        dot(destination.s.xyz, source.s.xyz),
        dot(destination.p.xyz, source.s.xyz),
        dot(destination.s.xyz, source.p.xyz),
        dot(destination.p.xyz, source.p.xyz)
    );
}

float complex_phase_space_determinant(ComplexOpticalOperator op) {
    return determinant(op.phase_space);
}

#endif
