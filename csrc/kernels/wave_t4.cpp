/**
 * wave_t4.cpp — Stateful T4 CPU-side field ownership helpers.
 *
 * Numerical propagation backends remain separate, but all of them use the
 * exact-band selection and boundary accounting defined here.
 */
#include "wave_t4.h"
#include "plan_support.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace wave_t4 {
namespace {

using FftMatrix = Eigen::Matrix<std::complex<float>,
                                Eigen::Dynamic,
                                Eigen::Dynamic,
                                Eigen::ColMajor>;

struct FftfreeExecutor {
    eigfft::PlanEnvironment<float> forward_x;
    eigfft::PlanEnvironment<float> forward_y;
    eigfft::PlanEnvironment<float> inverse_x;
    eigfft::PlanEnvironment<float> inverse_y;
    std::vector<std::complex<float>> interleaved;

    FftfreeExecutor(int nx, int ny)
        : interleaved(static_cast<std::size_t>(nx) * ny)
    {
        eigfft::PlanRuntimeConfig config;
        config.threads = 1;
        config.lanes =
            eigfft::Plan<float>::Limits::compile_time_max_lane_capacity();
        config.transpose_capacity = interleaved.size();
        config.allow_outer_parallel = false;
        config.allow_inner_parallel = false;
        config.transform = 0;
        forward_x.initialize(nx, false, config);
        forward_y.initialize(ny, false, config);
        inverse_x.initialize(nx, true, config);
        inverse_y.initialize(ny, true, config);
        forward_x.plan().use_kernel(eigfft::KernelKind::CooleyTukey);
        forward_y.plan().use_kernel(eigfft::KernelKind::CooleyTukey);
        inverse_x.plan().use_kernel(eigfft::KernelKind::CooleyTukey);
        inverse_y.plan().use_kernel(eigfft::KernelKind::CooleyTukey);
    }

    void transform(float* re, float* im, int nx, int ny, bool inverse)
    {
        const std::size_t count = static_cast<std::size_t>(nx) * ny;
        for (std::size_t i = 0; i < count; ++i)
            interleaved[i] = {re[i], im[i]};
        Eigen::Map<FftMatrix> matrix(interleaved.data(), nx, ny);
        if (inverse)
            eigfft::fft_inplace_2d<float>(
                matrix, inverse_x.plan(), inverse_y.plan());
        else
            eigfft::fft_inplace_2d<float>(
                matrix, forward_x.plan(), forward_y.plan());
        for (std::size_t i = 0; i < count; ++i) {
            re[i] = interleaved[i].real();
            im[i] = interleaved[i].imag();
        }
    }
};

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

template <int B, bool UseFftfree>
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
    FftfreeExecutor* executor = nullptr;
    if constexpr (UseFftfree) {
        executor = static_cast<FftfreeExecutor*>(
            plan->transform_executor.get());
        if (!executor) return false;
    }
    for (int band = 0; band < B; ++band) {
        const double wavelength = wavelengths_m[band];
        if (!(wavelength > 0.0)) continue;
        float* band_re = re + static_cast<std::size_t>(band) * plane;
        float* band_im = im + static_cast<std::size_t>(band) * plane;
        if constexpr (UseFftfree) {
            try {
                executor->transform(band_re, band_im, nx, ny, false);
            } catch (...) {
                return false;
            }
        } else {
            fft_2d(band_re, band_im, *plan, false);
        }

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
        if constexpr (UseFftfree) {
            try {
                executor->transform(band_re, band_im, nx, ny, true);
            } catch (...) {
                return false;
            }
        } else {
            fft_2d(band_re, band_im, *plan, true);
        }
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

inline bool aperture_open(const ApertureMaterial& material,
                          double x,
                          double y) noexcept
{
    const double c = std::cos(material.rotation_rad);
    const double s = std::sin(material.rotation_rad);
    const double u = c*x + s*y;
    const double v = -s*x + c*y;
    switch (material.pattern) {
        case AperturePattern::IrisPolygon: {
            if (material.element_count < 3 || material.opening_x_m <= 0.0)
                return false;
            const double radius = material.opening_x_m;
            for (int edge = 0; edge < material.element_count; ++edge) {
                const double a0 =
                    6.2831853071795864769 * edge / material.element_count;
                const double a1 =
                    6.2831853071795864769 * (edge + 1)
                    / material.element_count;
                const double x0 = radius * std::cos(a0);
                const double y0 = radius * std::sin(a0);
                const double x1 = radius * std::cos(a1);
                const double y1 = radius * std::sin(a1);
                if ((x1-x0)*(v-y0) - (y1-y0)*(u-x0) < 0.0)
                    return false;
            }
            return true;
        }
        case AperturePattern::ShadowMask: {
            if (material.pitch_x_m <= 0.0 || material.pitch_y_m <= 0.0
                || material.opening_x_m <= 0.0
                || material.opening_y_m <= 0.0)
                return false;
            const double cell_x =
                u - std::round(u/material.pitch_x_m)*material.pitch_x_m;
            const double cell_y =
                v - std::round(v/material.pitch_y_m)*material.pitch_y_m;
            const double ex = cell_x/material.opening_x_m;
            const double ey = cell_y/material.opening_y_m;
            return ex*ex + ey*ey <= 1.0;
        }
        case AperturePattern::SlotMask: {
            if (material.pitch_x_m <= 0.0 || material.pitch_y_m <= 0.0)
                return false;
            const double cell_x =
                u - std::round(u/material.pitch_x_m)*material.pitch_x_m;
            const double cell_y =
                v - std::round(v/material.pitch_y_m)*material.pitch_y_m;
            return std::abs(cell_x) <= material.opening_x_m
                && std::abs(cell_y) <= material.opening_y_m;
        }
        case AperturePattern::ApertureGrille: {
            if (material.pitch_x_m <= 0.0) return false;
            const double cell_x =
                u - std::round(u/material.pitch_x_m)*material.pitch_x_m;
            return std::abs(cell_x) <= material.opening_x_m;
        }
        case AperturePattern::None:
        default:
            return true;
    }
}

template <int B>
bool aperture_material_exact(int nx,
                             int ny,
                             double dx,
                             const double* wavelengths_m,
                             int direction_sign,
                             const ApertureMaterial& material,
                             double distance_m,
                             float* re,
                             float* im,
                             Progress* progress) noexcept
{
    if (!re || !im || !wavelengths_m || !progress || nx <= 0 || ny <= 0
        || !(dx > 0.0) || !(distance_m > 0.0)
        || (direction_sign != 1 && direction_sign != -1)
        || material.pattern == AperturePattern::None
        || material.assembly_radius_m <= 0.0
        || material.material_n_imag < 0.0)
        return false;
    constexpr double tau = 6.283185307179586476925286766559;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const double origin_x = 0.5 * static_cast<double>(nx - 1) * dx;
    const double origin_y = 0.5 * static_cast<double>(ny - 1) * dx;
    const double assembly_r2 =
        material.assembly_radius_m * material.assembly_radius_m;
    double removed = 0.0;
    for (int y = 0; y < ny; ++y) {
        const double ym = y*dx - origin_y;
        for (int x = 0; x < nx; ++x) {
            const double xm = x*dx - origin_x;
            if (xm*xm + ym*ym > assembly_r2
                || aperture_open(material, xm, ym))
                continue;
            const std::size_t pixel = static_cast<std::size_t>(y)*nx + x;
            for (int band = 0; band < B; ++band) {
                const double wavelength = wavelengths_m[band];
                if (!(wavelength > 0.0)) continue;
                const double k0 = tau/wavelength;
                const double phase = direction_sign * k0
                    * (material.material_n_real-material.background_n_real)
                    * distance_m;
                const double attenuation = std::exp(
                    -k0*material.material_n_imag*distance_m);
                const double tr = attenuation*std::cos(phase);
                const double ti = attenuation*std::sin(phase);
                const std::size_t i =
                    static_cast<std::size_t>(band)*plane + pixel;
                const double ar = re[i];
                const double ai = im[i];
                const double before = ar*ar + ai*ai;
                const double nr = ar*tr - ai*ti;
                const double ni = ar*ti + ai*tr;
                re[i] = static_cast<float>(nr);
                im[i] = static_cast<float>(ni);
                removed += std::max(0.0, before-(nr*nr+ni*ni));
            }
        }
    }
    progress->absorbed_power += removed;
    return true;
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
        plan->transform_executor =
            std::make_shared<FftfreeExecutor>(nx, ny);
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
#define WAVE_AS_CASE(B) case B: return angular_spectrum_exact<B, true>( \
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

bool angular_spectrum_step_reference(int bands,
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
#define WAVE_AS_REF_CASE(B) case B: return angular_spectrum_exact<B, false>( \
    nx, ny, dx, dz, wavelengths_m, direction_sign, plan, re, im)
    switch (bands) {
        WAVE_AS_REF_CASE(1);
        WAVE_AS_REF_CASE(3);
        WAVE_AS_REF_CASE(4);
        WAVE_AS_REF_CASE(8);
        WAVE_AS_REF_CASE(16);
        WAVE_AS_REF_CASE(32);
        default: return false;
    }
#undef WAVE_AS_REF_CASE
}

bool apply_aperture_material(int bands,
                             int nx,
                             int ny,
                             double dx,
                             const double* wavelengths_m,
                             int direction_sign,
                             const ApertureMaterial& material,
                             double distance_m,
                             float* re,
                             float* im,
                             Progress* progress) noexcept
{
#define WAVE_APERTURE_CASE(B) case B: return aperture_material_exact<B>( \
    nx, ny, dx, wavelengths_m, direction_sign, material, distance_m, \
    re, im, progress)
    switch (bands) {
        WAVE_APERTURE_CASE(1);
        WAVE_APERTURE_CASE(3);
        WAVE_APERTURE_CASE(4);
        WAVE_APERTURE_CASE(8);
        WAVE_APERTURE_CASE(16);
        WAVE_APERTURE_CASE(32);
        default: return false;
    }
#undef WAVE_APERTURE_CASE
}

}  // namespace wave_t4
