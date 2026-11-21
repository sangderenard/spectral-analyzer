#include "wave_t4_c.h"
#include "wave_t4.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

struct WaveT4BatchPlan {
    wave_t4::AngularSpectrumPlan angular;
    std::vector<float> real;
    std::vector<float> imag;
};

WaveT4BatchPlan* wave_t4_batch_plan_create(int nx, int ny)
{
    auto* plan = new (std::nothrow) WaveT4BatchPlan();
    if (!plan) return nullptr;
    if (!wave_t4::build_angular_spectrum_plan(nx, ny, &plan->angular)) {
        delete plan;
        return nullptr;
    }
    return plan;
}

void wave_t4_batch_plan_destroy(WaveT4BatchPlan* plan) { delete plan; }

int wave_t4_batch_angular_step(
    WaveT4BatchPlan* plan, int arena_count, int bands,
    double dx, double dz, int direction_sign,
    const double* wavelengths_m,
    float* const* real_planes, float* const* imag_planes)
{
    if (!plan || !wavelengths_m || !real_planes || !imag_planes
        || arena_count < 1 || bands < 1 || arena_count > 32
        || bands > 32 || arena_count * bands > 32
        || !(std::isfinite(dx) && dx > 0.0)
        || !(std::isfinite(dz) && dz != 0.0)
        || (direction_sign != 1 && direction_sign != -1))
        return 1;
    const auto nx = plan->angular.nx, ny = plan->angular.ny;
    if (nx <= 0 || ny <= 0) return 1;
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t per_arena = plane * static_cast<size_t>(bands);
    const size_t total = per_arena * static_cast<size_t>(arena_count);
    if (total > (512ull << 20) / sizeof(float) / 2u) return 1;
    for (int lane = 0; lane < arena_count * bands; ++lane)
        if (!(std::isfinite(wavelengths_m[lane]) && wavelengths_m[lane] > 0.0))
            return 1;
    for (int arena = 0; arena < arena_count; ++arena)
        if (!real_planes[arena] || !imag_planes[arena]) return 1;
    if (arena_count == 1)
        return wave_t4::angular_spectrum_step_wide(
            bands, nx, ny, dx, dz, wavelengths_m, direction_sign,
            &plan->angular, real_planes[0], imag_planes[0]) ? 0 : 2;
    try {
        plan->real.resize(total);
        plan->imag.resize(total);
    } catch (...) { return 2; }
    for (int arena = 0; arena < arena_count; ++arena) {
        const size_t offset = static_cast<size_t>(arena) * per_arena;
        std::memcpy(plan->real.data() + offset, real_planes[arena], per_arena*sizeof(float));
        std::memcpy(plan->imag.data() + offset, imag_planes[arena], per_arena*sizeof(float));
    }
    if (!wave_t4::angular_spectrum_step_wide(
            arena_count * bands, nx, ny, dx, dz, wavelengths_m,
            direction_sign, &plan->angular,
            plan->real.data(), plan->imag.data())) return 2;
    for (int arena = 0; arena < arena_count; ++arena) {
        const size_t offset = static_cast<size_t>(arena) * per_arena;
        std::memcpy(real_planes[arena], plan->real.data() + offset, per_arena*sizeof(float));
        std::memcpy(imag_planes[arena], plan->imag.data() + offset, per_arena*sizeof(float));
    }
    return 0;
}
