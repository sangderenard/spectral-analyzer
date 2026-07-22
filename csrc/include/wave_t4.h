/**
 * wave_t4.h — Stable ownership and specialization contract for optical T4.
 *
 * Production T4 fields are compiled for an exact spectral lane count.  The
 * selected specialization owns only the memory required by that count; an
 * unsupported count is rejected instead of being rounded up to a wider ABI.
 * The production numerical target is the bidirectional two-component vector
 * angular-spectrum engine specified by WAVE_ENGINE_ACTION_PLAN.md. Localized
 * full-Maxwell patches are compiled separately according to
 * MAXWELL_PATCH_CONTEXT.md and enter T4 as scattering artifacts.
 */
#pragma once

#include <array>
#include <cstddef>

namespace wave_t4 {

inline constexpr std::array<int, 6> kBandCounts = {1, 3, 4, 8, 16, 32};

inline constexpr int specialization_index(int bands) noexcept
{
    for (std::size_t i = 0; i < kBandCounts.size(); ++i)
        if (kBandCounts[i] == bands) return static_cast<int>(i);
    return -1;
}

inline constexpr bool supports_band_count(int bands) noexcept
{
    return specialization_index(bands) >= 0;
}

enum class SpectralMode : int {
    FixedBands = 0,
    ContinuousCohort = 1,
};

/* T4 is an execution/ownership slot, not the name of the current numerical
 * method. Backends can change without changing the packed-state contract. */
enum class BackendKind : int {
    LegacyAdiCalibration = 0,
    AngularSpectrum = 1,
};

struct SpectralLane {
    double frequency_hz = 0.0;
    double wavelength_m = 0.0;
    float pdf = 1.0f;
    unsigned int coherence_id = 0;
    unsigned int active = 0;
};

struct BoundaryConfig {
    int guard_cells = 16;
    int absorber_cells = 12;
    float absorber_strength = 8.0f;
};

struct Progress {
    unsigned long long generation = 0;
    unsigned long long completed_steps = 0;
    double input_power = 0.0;
    double field_power = 0.0;
    double border_power = 0.0;
    double absorbed_power = 0.0;
};

/** Apply the numerical exterior absorber in-place using an exact-band CPU
 * specialization. Returns false for unsupported band counts or invalid input. */
bool apply_absorbing_border(int bands,
                            int nx,
                            int ny,
                            const BoundaryConfig& config,
                            float step_fraction,
                            float* re,
                            float* im,
                            Progress* progress) noexcept;

/** Measure field and border power without modifying the field. */
bool measure(int bands,
             int nx,
             int ny,
             const BoundaryConfig& config,
             const float* re,
             const float* im,
             Progress* progress) noexcept;

/** One legacy ADI calibration step using only caller-owned persistent state.
 * Every pointer is a fixed view into the arena's contiguous state block. */
bool legacy_adi_step(int bands,
                     int nx,
                     int ny,
                     float dx,
                     float dz,
                     const double* wavelengths_m,
                     float* re,
                     float* im,
                     float* tmp_re,
                     float* tmp_im,
                     float* rhs_re,
                     float* rhs_im,
                     float* cp_re,
                     float* cp_im,
                     float* dp_re,
                     float* dp_im) noexcept;

}  // namespace wave_t4
