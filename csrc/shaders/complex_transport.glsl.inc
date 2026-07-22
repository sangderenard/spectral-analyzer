/* complex_transport.glsl.inc -- GLSL half of complex_transport.h schema v1.
 *
 * This include deliberately declares no SSBO binding.  Producing stages can
 * write these 16-word records into a tail region of an already-bound buffer;
 * T4 may later bind that same storage under its own binding map.
 */
#ifndef COMPLEX_TRANSPORT_GLSL_INC
#define COMPLEX_TRANSPORT_GLSL_INC

#define COMPLEX_SCHEMA_VERSION       1u
#define COMPLEX_LANE_WORDS          16u
#define COMPLEX_LANE_BYTES          64u

#define COMPLEX_FREQUENCY_HI         0u
#define COMPLEX_FREQUENCY_LO         1u
#define COMPLEX_OPTICAL_PATH_HI      2u
#define COMPLEX_OPTICAL_PATH_LO      3u
#define COMPLEX_SPECTRAL_PDF         4u
#define COMPLEX_TRANSPORT_JACOBIAN   5u
#define COMPLEX_AMP_S_RE             6u
#define COMPLEX_AMP_S_IM             7u
#define COMPLEX_AMP_P_RE             8u
#define COMPLEX_AMP_P_IM             9u
#define COMPLEX_COHERENCE_LO        10u
#define COMPLEX_COHERENCE_HI        11u
#define COMPLEX_SAMPLE_ID           12u
#define COMPLEX_LANE_FLAGS          13u
#define COMPLEX_RESERVED_0          14u
#define COMPLEX_RESERVED_1          15u

#define COMPLEX_FLAG_ACTIVE          (1u << 0u)
#define COMPLEX_FLAG_CONTINUOUS      (1u << 1u)
#define COMPLEX_FLAG_COHERENT        (1u << 2u)
#define COMPLEX_FLAG_JONES_VALID     (1u << 3u)
#define COMPLEX_FLAG_OPL_VALID       (1u << 4u)
#define COMPLEX_FLAG_JACOBIAN_VALID  (1u << 5u)

uint complex_lane_base(uint block_base_words, uint lane) {
    return block_base_words + lane * COMPLEX_LANE_WORDS;
}

uint complex_lane_flags(uint packed) { return packed & 0xffffu; }
uint complex_source_lane(uint packed) { return packed >> 16u; }

/* Float-float values remain split unless the consuming shader has fp64.  For
 * local float calculations, hi+lo is the best available reconstruction. */
float complex_join_ff(float hi, float lo) { return hi + lo; }

#endif
