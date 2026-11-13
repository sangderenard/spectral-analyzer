#pragma once

#include <stdint.h>

#ifdef _WIN32
# ifdef BUILDING_WAVE_T4_DLL
#  define WAVE_T4_API __declspec(dllexport)
# else
#  define WAVE_T4_API __declspec(dllimport)
# endif
#else
# define WAVE_T4_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WaveT4BatchPlan WaveT4BatchPlan;

/* One owner per Nodus row. The fftfree plan is not concurrently reentered. */
WAVE_T4_API WaveT4BatchPlan* wave_t4_batch_plan_create(int nx, int ny);
WAVE_T4_API void wave_t4_batch_plan_destroy(WaveT4BatchPlan* plan);

/* One genuine T4 angular-spectrum step across compatible, disjoint arenas.
 * Each arena has `bands` contiguous planes of nx*ny float samples. The
 * wavelength array is [arena][band]; all arenas share dx, dz and direction.
 * The caller retains each arena's field in its tensor-library address.
 * Returns 0 on success; otherwise no destination plane is changed. */
WAVE_T4_API int wave_t4_batch_angular_step(
    WaveT4BatchPlan* plan, int arena_count, int bands,
    double dx, double dz, int direction_sign,
    const double* wavelengths_m,
    float* const* real_planes, float* const* imag_planes);

#ifdef __cplusplus
}
#endif
