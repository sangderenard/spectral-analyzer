/**
 * wave_t4.cpp — Stateful T4 CPU-side field ownership helpers.
 *
 * Numerical propagation backends remain separate, but all of them use the
 * exact-band selection and boundary accounting defined here.
 */
#include "wave_t4.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace wave_t4 {
namespace {

inline bool is_power_of_two(int value) noexcept
{
    return value > 0 && (value & (value - 1)) == 0;
}

inline void swap_complex(float* re, float* im,
                         std::size_t a, std::size_t b) noexcept
{
    std::swap(re[a], re[b]);
    std::swap(im[a], im[b]);
}

void fft_strided(float* re,
                 float* im,
                 std::size_t base,
                 std::size_t stride,
                 const FftAxisPlan& plan,
                 bool inverse) noexcept
{
    const int count = plan.size;
    for (int i = 1; i < count; ++i) {
        const int j = static_cast<int>(
            plan.bit_reverse[static_cast<std::size_t>(i)]);
        if (i < j)
            swap_complex(re, im,
                         base + static_cast<std::size_t>(i) * stride,
                         base + static_cast<std::size_t>(j) * stride);
    }

    for (int length = 2; length <= count; length <<= 1) {
        const int half = length >> 1;
        const int root_stride = count / length;
        for (int start = 0; start < count; start += length) {
            for (int j = 0; j < half; ++j) {
                const std::size_t root =
                    static_cast<std::size_t>(j * root_stride);
                const double wr = plan.root_re[root];
                const double wi = inverse
                    ? -static_cast<double>(plan.root_im[root])
                    : static_cast<double>(plan.root_im[root]);
                const std::size_t even_i =
                    base + static_cast<std::size_t>(start + j) * stride;
                const std::size_t odd_i =
                    base + static_cast<std::size_t>(start + j + half) * stride;
                const double odd_re = re[odd_i] * wr - im[odd_i] * wi;
                const double odd_im = re[odd_i] * wi + im[odd_i] * wr;
                const double even_re = re[even_i];
                const double even_im = im[even_i];
                re[even_i] = static_cast<float>(even_re + odd_re);
                im[even_i] = static_cast<float>(even_im + odd_im);
                re[odd_i] = static_cast<float>(even_re - odd_re);
                im[odd_i] = static_cast<float>(even_im - odd_im);
            }
        }
    }

    if (inverse) {
        const float scale = 1.0f / static_cast<float>(count);
        for (int i = 0; i < count; ++i) {
            const std::size_t index =
                base + static_cast<std::size_t>(i) * stride;
            re[index] *= scale;
            im[index] *= scale;
        }
    }
}

void fft_2d(float* re,
            float* im,
            const AngularSpectrumPlan& plan,
            bool inverse) noexcept
{
    const int nx = plan.nx;
    const int ny = plan.ny;
    for (int y = 0; y < ny; ++y)
        fft_strided(re, im, static_cast<std::size_t>(y) * nx,
                    1u, plan.x, inverse);
    for (int x = 0; x < nx; ++x)
        fft_strided(re, im, static_cast<std::size_t>(x),
                    static_cast<std::size_t>(nx), plan.y, inverse);
}

template <int B>
bool angular_spectrum_exact(int nx,
                            int ny,
                            double dx,
                            double dz,
                            const double* wavelengths_m,
                            int direction_sign,
                            const AngularSpectrumPlan* plan,
                            float* re,
                            float* im) noexcept
{
    if (!re || !im || !wavelengths_m || !is_power_of_two(nx)
        || !is_power_of_two(ny) || !(dx > 0.0) || dz == 0.0
        || (direction_sign != 1 && direction_sign != -1)
        || !plan || plan->nx != nx || plan->ny != ny)
        return false;

    constexpr double tau = 6.283185307179586476925286766559;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const double signed_distance = direction_sign * dz;
    for (int band = 0; band < B; ++band) {
        const double wavelength = wavelengths_m[band];
        if (!(wavelength > 0.0)) continue;
        float* band_re = re + static_cast<std::size_t>(band) * plane;
        float* band_im = im + static_cast<std::size_t>(band) * plane;
        fft_2d(band_re, band_im, *plan, false);

        const double k = tau / wavelength;
        for (int y = 0; y < ny; ++y) {
            const int fy = y <= ny / 2 ? y : y - ny;
            const double ky = tau * fy / (static_cast<double>(ny) * dx);
            for (int x = 0; x < nx; ++x) {
                const int fx = x <= nx / 2 ? x : x - nx;
                const double kx = tau * fx / (static_cast<double>(nx) * dx);
                const double kz2 = k * k - kx * kx - ky * ky;
                double transfer_re = 0.0;
                double transfer_im = 0.0;
                if (kz2 >= 0.0) {
                    const double phase = signed_distance * std::sqrt(kz2);
                    transfer_re = std::cos(phase);
                    transfer_im = std::sin(phase);
                } else {
                    const double decay =
                        std::exp(-std::abs(dz) * std::sqrt(-kz2));
                    transfer_re = decay;
                }
                const std::size_t index =
                    static_cast<std::size_t>(y) * nx + x;
                const double ar = band_re[index];
                const double ai = band_im[index];
                band_re[index] =
                    static_cast<float>(ar * transfer_re - ai * transfer_im);
                band_im[index] =
                    static_cast<float>(ar * transfer_im + ai * transfer_re);
            }
        }
        fft_2d(band_re, band_im, *plan, true);
    }
    return true;
}

template <int B>
bool measure_exact(int nx,
                   int ny,
                   const BoundaryConfig& config,
                   const float* re,
                   const float* im,
                   Progress* progress) noexcept
{
    if (!re || !im || !progress || nx <= 0 || ny <= 0) return false;
    const int edge = std::max(0, std::min(config.absorber_cells,
                                          std::min(nx, ny) / 2));
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    double total = 0.0;
    double border = 0.0;
    for (int b = 0; b < B; ++b) {
        const std::size_t band = static_cast<std::size_t>(b) * plane;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const std::size_t i = band + static_cast<std::size_t>(y) * nx + x;
                const double p = static_cast<double>(re[i]) * re[i]
                               + static_cast<double>(im[i]) * im[i];
                total += p;
                if (x < edge || x >= nx - edge || y < edge || y >= ny - edge)
                    border += p;
            }
        }
    }
    progress->field_power = total;
    progress->border_power = border;
    return true;
}

template <int B>
bool absorb_exact(int nx,
                  int ny,
                  const BoundaryConfig& config,
                  float step_fraction,
                  float* re,
                  float* im,
                  Progress* progress) noexcept
{
    if (!re || !im || !progress || nx <= 0 || ny <= 0) return false;
    const int edge = std::max(0, std::min(config.absorber_cells,
                                          std::min(nx, ny) / 2));
    if (edge == 0 || !(config.absorber_strength > 0.0f))
        return measure_exact<B>(nx, ny, config, re, im, progress);

    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    double removed = 0.0;
    const float fraction = std::max(0.0f, step_fraction);
    for (int y = 0; y < ny; ++y) {
        const int dy = std::min(y, ny - 1 - y);
        for (int x = 0; x < nx; ++x) {
            const int dx = std::min(x, nx - 1 - x);
            const int distance = std::min(dx, dy);
            if (distance >= edge) continue;
            const float u = static_cast<float>(edge - distance)
                          / static_cast<float>(edge);
            const float u2 = u * u;
            const float attenuation = std::exp(
                -config.absorber_strength * fraction * u2 * u2);
            for (int b = 0; b < B; ++b) {
                const std::size_t i = static_cast<std::size_t>(b) * plane
                                    + static_cast<std::size_t>(y) * nx + x;
                const double before = static_cast<double>(re[i]) * re[i]
                                    + static_cast<double>(im[i]) * im[i];
                re[i] *= attenuation;
                im[i] *= attenuation;
                const double after = static_cast<double>(re[i]) * re[i]
                                   + static_cast<double>(im[i]) * im[i];
                removed += std::max(0.0, before - after);
            }
        }
    }
    progress->absorbed_power += removed;
    ++progress->completed_steps;
    return measure_exact<B>(nx, ny, config, re, im, progress);
}

}  // namespace

bool build_angular_spectrum_plan(int nx,
                                 int ny,
                                 AngularSpectrumPlan* plan) noexcept
{
    if (!plan || !is_power_of_two(nx) || !is_power_of_two(ny))
        return false;
    auto build_axis = [](int size, FftAxisPlan& axis) {
        axis.size = size;
        axis.bit_reverse.resize(static_cast<std::size_t>(size));
        axis.root_re.resize(static_cast<std::size_t>(size / 2));
        axis.root_im.resize(static_cast<std::size_t>(size / 2));
        int bits = 0;
        while ((1 << bits) < size) ++bits;
        for (int value = 0; value < size; ++value) {
            std::uint32_t source = static_cast<std::uint32_t>(value);
            std::uint32_t reversed = 0u;
            for (int bit = 0; bit < bits; ++bit) {
                reversed = (reversed << 1u) | (source & 1u);
                source >>= 1u;
            }
            axis.bit_reverse[static_cast<std::size_t>(value)] = reversed;
        }
        constexpr double tau = 6.283185307179586476925286766559;
        for (int root = 0; root < size / 2; ++root) {
            const double angle = -tau * root / size;
            axis.root_re[static_cast<std::size_t>(root)] =
                static_cast<float>(std::cos(angle));
            axis.root_im[static_cast<std::size_t>(root)] =
                static_cast<float>(std::sin(angle));
        }
    };
    try {
        plan->nx = nx;
        plan->ny = ny;
        build_axis(nx, plan->x);
        build_axis(ny, plan->y);
    } catch (...) {
        *plan = {};
        return false;
    }
    return true;
}

bool apply_absorbing_border(int bands,
                            int nx,
                            int ny,
                            const BoundaryConfig& config,
                            float step_fraction,
                            float* re,
                            float* im,
                            Progress* progress) noexcept
{
    switch (bands) {
        case 1:  return absorb_exact<1>(nx, ny, config, step_fraction, re, im, progress);
        case 3:  return absorb_exact<3>(nx, ny, config, step_fraction, re, im, progress);
        case 4:  return absorb_exact<4>(nx, ny, config, step_fraction, re, im, progress);
        case 8:  return absorb_exact<8>(nx, ny, config, step_fraction, re, im, progress);
        case 16: return absorb_exact<16>(nx, ny, config, step_fraction, re, im, progress);
        case 32: return absorb_exact<32>(nx, ny, config, step_fraction, re, im, progress);
        default: return false;
    }
}

bool measure(int bands,
             int nx,
             int ny,
             const BoundaryConfig& config,
             const float* re,
             const float* im,
             Progress* progress) noexcept
{
    switch (bands) {
        case 1:  return measure_exact<1>(nx, ny, config, re, im, progress);
        case 3:  return measure_exact<3>(nx, ny, config, re, im, progress);
        case 4:  return measure_exact<4>(nx, ny, config, re, im, progress);
        case 8:  return measure_exact<8>(nx, ny, config, re, im, progress);
        case 16: return measure_exact<16>(nx, ny, config, re, im, progress);
        case 32: return measure_exact<32>(nx, ny, config, re, im, progress);
        default: return false;
    }
}

bool angular_spectrum_step(int bands,
                           int nx,
                           int ny,
                           double dx,
                           double dz,
                           const double* wavelengths_m,
                           int direction_sign,
                           const AngularSpectrumPlan* plan,
                           float* re,
                           float* im) noexcept
{
#define WAVE_AS_CASE(B) case B: return angular_spectrum_exact<B>( \
    nx, ny, dx, dz, wavelengths_m, direction_sign, plan, re, im)
    switch (bands) {
        WAVE_AS_CASE(1);
        WAVE_AS_CASE(3);
        WAVE_AS_CASE(4);
        WAVE_AS_CASE(8);
        WAVE_AS_CASE(16);
        WAVE_AS_CASE(32);
        default: return false;
    }
#undef WAVE_AS_CASE
}

}  // namespace wave_t4
