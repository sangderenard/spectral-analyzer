/**
 * wave_t4.cpp — Stateful T4 CPU-side field ownership helpers.
 *
 * Numerical propagation backends remain separate, but all of them use the
 * exact-band selection and boundary accounting defined here.
 */
#include "wave_t4.h"
#include "plan_support.hpp"

#include <algorithm>
#include <array>
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

    /* Angular-spectrum transfer function H(kx,ky) for one (|dz|, wavelength)
     * pair. H never changes for a given arena geometry, yet the previous
     * implementation recomputed sqrt+cos+sin for every pixel of every band
     * on every substep of every march (~tens of millions of libm calls per
     * traversal -- the measured T4 hot spot). Built once here with the
     * identical double-precision formula, each step becomes FFT + one
     * elementwise complex multiply. Backward propagation is the conjugate:
     * the sign multiplies t_sin at apply time. Evanescent components decay
     * by exp(-|dz|sqrt(-kz2)) regardless of direction, stored in t_cos with
     * t_sin = 0, exactly as the analytic loop produced. */
    struct TransferTable {
        double distance = -1.0;
        double wavelength = -1.0;
        std::uint64_t last_use = 0;
        std::vector<double> t_cos;
        std::vector<double> t_sin;
    };
    std::vector<TransferTable> transfer_tables;
    std::uint64_t transfer_clock = 0;

    const TransferTable& transfer_for(
        int nx, int ny, double dx, double abs_dz, double wavelength)
    {
        ++transfer_clock;
        for (auto& table : transfer_tables) {
            if (table.distance == abs_dz
                && table.wavelength == wavelength) {
                table.last_use = transfer_clock;
                return table;
            }
        }
        TransferTable* slot;
        /* Fixed-band arenas use a handful of (distance, wavelength) pairs;
         * continuous cohorts resolve fresh wavelengths per round, so the
         * cache is bounded and recycles its least-recently-used entry. */
        if (transfer_tables.size() < 64u) {
            transfer_tables.emplace_back();
            slot = &transfer_tables.back();
        } else {
            slot = &*std::min_element(
                transfer_tables.begin(), transfer_tables.end(),
                [](const TransferTable& a, const TransferTable& b) {
                    return a.last_use < b.last_use;
                });
        }
        constexpr double tau = 6.283185307179586476925286766559;
        const std::size_t plane =
            static_cast<std::size_t>(nx) * ny;
        slot->distance = abs_dz;
        slot->wavelength = wavelength;
        slot->last_use = transfer_clock;
        slot->t_cos.resize(plane);
        slot->t_sin.resize(plane);
        const double k = tau / wavelength;
        for (int y = 0; y < ny; ++y) {
            const int fy = y <= ny / 2 ? y : y - ny;
            const double ky = tau * fy / (static_cast<double>(ny) * dx);
            for (int x = 0; x < nx; ++x) {
                const int fx = x <= nx / 2 ? x : x - nx;
                const double kx = tau * fx / (static_cast<double>(nx) * dx);
                const double kz2 = k * k - kx * kx - ky * ky;
                const std::size_t index =
                    static_cast<std::size_t>(y) * nx + x;
                if (kz2 >= 0.0) {
                    const double phase = abs_dz * std::sqrt(kz2);
                    slot->t_cos[index] = std::cos(phase);
                    slot->t_sin[index] = std::sin(phase);
                } else {
                    slot->t_cos[index] =
                        std::exp(-abs_dz * std::sqrt(-kz2));
                    slot->t_sin[index] = 0.0;
                }
            }
        }
        return *slot;
    }

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

    /* Transform `n_bands` contiguous (nx, ny) planes in re/im with two
     * batched axis passes instead of n_bands separate 2D FFT calls.
     *
     * Each band's plane is itself a (nx rows, ny cols) ColMajor block, and
     * bands are stored contiguously (band stride = nx*ny), so the whole
     * re/im buffer is already a valid (nx, ny*n_bands) ColMajor matrix with
     * no reshaping: column b*ny+j is band b's column j. The axis-0 (row)
     * pass can therefore batch across every band's columns in one
     * fft_inplace_batched call. The transpose step must stay per-band
     * (transposing the whole (nx, ny*n_bands) block as one unit would
     * scramble band boundaries), but transposition is a memory-bound copy,
     * not FFT work, so looping it costs nothing like the FFT calls did. */
    void transform_multiband(
        float* re, float* im, int nx, int ny, int n_bands, bool inverse)
    {
        using Complex = std::complex<float>;
        const std::size_t plane = static_cast<std::size_t>(nx) * ny;
        const std::size_t total = plane * static_cast<std::size_t>(n_bands);
        if (interleaved.size() < total) interleaved.resize(total);
        for (std::size_t i = 0; i < total; ++i)
            interleaved[i] = {re[i], im[i]};

        auto& axis0 = inverse ? inverse_x.plan() : forward_x.plan();
        auto& axis1 = inverse ? inverse_y.plan() : forward_y.plan();

        Eigen::Map<FftMatrix> wide(
            interleaved.data(), nx,
            static_cast<Eigen::Index>(ny) * n_bands);
        eigfft::fft_inplace_batched<float>(wide, axis0);

        axis0.ensure_nd_workspace(
            nx, static_cast<Eigen::Index>(ny) * n_bands);
        Complex* scratch = axis0.transpose_buffer_data();
        for (int b = 0; b < n_bands; ++b)
            eigfft::detail::tiled_transpose_colmajor<float>(
                interleaved.data() + static_cast<std::size_t>(b) * plane,
                nx, ny, scratch + static_cast<std::size_t>(b) * plane,
                64, 128);

        Eigen::Map<FftMatrix> wide_t(
            scratch, ny, static_cast<Eigen::Index>(nx) * n_bands);
        eigfft::fft_inplace_batched<float>(wide_t, axis1);

        for (int b = 0; b < n_bands; ++b)
            eigfft::detail::tiled_transpose_colmajor<float>(
                scratch + static_cast<std::size_t>(b) * plane,
                ny, nx, interleaved.data() + static_cast<std::size_t>(b) * plane,
                64, 128);

        for (std::size_t i = 0; i < total; ++i) {
            re[i] = interleaved[i].real();
            im[i] = interleaved[i].imag();
        }
    }

    void transform(float* re, float* im, int nx, int ny, bool inverse)
    {
        transform_multiband(re, im, nx, ny, 1, inverse);
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
        /* One batched forward pass over all B bands instead of B separate
         * 2D FFT calls. Inactive bands (wavelength <= 0) still hold zeroed
         * field data (arenas are cleared at seed time), so transforming
         * them is harmless -- only the transfer-function multiply below,
         * which would divide by that zero wavelength, must skip them. */
        try {
            executor->transform_multiband(re, im, nx, ny, B, false);
        } catch (...) {
            return false;
        }
    }
    for (int band = 0; band < B; ++band) {
        const double wavelength = wavelengths_m[band];
        if (!(wavelength > 0.0)) continue;
        float* band_re = re + static_cast<std::size_t>(band) * plane;
        float* band_im = im + static_cast<std::size_t>(band) * plane;
        if constexpr (!UseFftfree) {
            fft_2d(band_re, band_im, *plan, false);
        }

        if constexpr (UseFftfree) {
            /* Precomputed transfer table: same double math, built once. */
            const auto& table = executor->transfer_for(
                nx, ny, dx, std::abs(dz), wavelength);
            const double dir = signed_distance >= 0.0 ? 1.0 : -1.0;
            for (std::size_t index = 0; index < plane; ++index) {
                const double transfer_re = table.t_cos[index];
                const double transfer_im = dir * table.t_sin[index];
                const double ar = band_re[index];
                const double ai = band_im[index];
                band_re[index] =
                    static_cast<float>(ar * transfer_re - ai * transfer_im);
                band_im[index] =
                    static_cast<float>(ar * transfer_im + ai * transfer_re);
            }
        } else {
            const double k = tau / wavelength;
            for (int y = 0; y < ny; ++y) {
                const int fy = y <= ny / 2 ? y : y - ny;
                const double ky = tau * fy / (static_cast<double>(ny) * dx);
                for (int x = 0; x < nx; ++x) {
                    const int fx = x <= nx / 2 ? x : x - nx;
                    const double kx =
                        tau * fx / (static_cast<double>(nx) * dx);
                    const double kz2 = k * k - kx * kx - ky * ky;
                    double transfer_re = 0.0;
                    double transfer_im = 0.0;
                    if (kz2 >= 0.0) {
                        const double phase =
                            signed_distance * std::sqrt(kz2);
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
                    band_re[index] = static_cast<float>(
                        ar * transfer_re - ai * transfer_im);
                    band_im[index] = static_cast<float>(
                        ar * transfer_im + ai * transfer_re);
                }
            }
        }
        if constexpr (!UseFftfree) {
            fft_2d(band_re, band_im, *plan, true);
        }
    }
    if constexpr (UseFftfree) {
        try {
            executor->transform_multiband(re, im, nx, ny, B, true);
        } catch (...) {
            return false;
        }
    }
    return true;
}

/* Same math as angular_spectrum_exact<B, true>, with a runtime band count
 * instead of a compile-time one. angular_spectrum_step()'s public dispatch
 * is restricted to the exact lane widths {1,3,4,8,16,32} -- a deliberate
 * ABI invariant for ordinary per-ray transport, not a limitation of the
 * math itself. A ray cohort's *combined* lane count (n_bands * ray count)
 * is not one ray's transport width and has no reason to obey that same
 * per-ray specialization set, so this is a separate entry point rather
 * than a relaxation of angular_spectrum_step's contract. Every lane still
 * gets exactly the same FFT + transfer-function math either way. */
bool angular_spectrum_wide(int bands,
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
    if (!re || !im || !wavelengths_m || bands <= 0 || !is_power_of_two(nx)
        || !is_power_of_two(ny) || !(dx > 0.0) || dz == 0.0
        || (direction_sign != 1 && direction_sign != -1)
        || !plan || plan->nx != nx || plan->ny != ny)
        return false;
    auto* executor = static_cast<FftfreeExecutor*>(
        plan->transform_executor.get());
    if (!executor) return false;

    constexpr double tau = 6.283185307179586476925286766559;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const double signed_distance = direction_sign * dz;
    try {
        executor->transform_multiband(re, im, nx, ny, bands, false);
    } catch (...) {
        return false;
    }
    const double dir = signed_distance >= 0.0 ? 1.0 : -1.0;
    for (int band = 0; band < bands; ++band) {
        const double wavelength = wavelengths_m[band];
        if (!(wavelength > 0.0)) continue;
        float* band_re = re + static_cast<std::size_t>(band) * plane;
        float* band_im = im + static_cast<std::size_t>(band) * plane;
        /* Precomputed transfer table: cohort lanes repeat a few distinct
         * wavelengths, so this is a cache hit for every lane after the
         * first ray of a round. */
        const auto& table = executor->transfer_for(
            nx, ny, dx, std::abs(dz), wavelength);
        for (std::size_t index = 0; index < plane; ++index) {
            const double transfer_re = table.t_cos[index];
            const double transfer_im = dir * table.t_sin[index];
            const double ar = band_re[index];
            const double ai = band_im[index];
            band_re[index] =
                static_cast<float>(ar * transfer_re - ai * transfer_im);
            band_im[index] =
                static_cast<float>(ar * transfer_im + ai * transfer_re);
        }
    }
    try {
        executor->transform_multiband(re, im, nx, ny, bands, true);
    } catch (...) {
        return false;
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

/* Per-pixel aperture test with everything that is constant across an entire
 * aperture_material_exact() call precomputed once. IrisPolygon in particular
 * was calling cos/sin ~4 times per edge inside aperture_open() for *every
 * pixel* (28 trig calls/pixel at element_count=7, ~1.8M total for a 256x256
 * grid, repeated for every active direction/component/substep) even though
 * the polygon vertices and rotation only depend on the material, not the
 * pixel. Precomputing the vertices (and the shared edge between consecutive
 * blades, computed once instead of twice) makes the per-pixel test pure
 * arithmetic. Vertex values are bit-identical to aperture_open()'s per-pixel
 * computation: same formula (2*pi*edge/n), same inputs, deterministic FP. */
struct AperturePixelTest {
    static constexpr int kMaxEdges = 64;
    double cos_rot = 1.0;
    double sin_rot = 0.0;
    AperturePattern pattern = AperturePattern::None;
    std::array<double, kMaxEdges> edge_x0{};
    std::array<double, kMaxEdges> edge_y0{};
    std::array<double, kMaxEdges> edge_x1{};
    std::array<double, kMaxEdges> edge_y1{};
    int edge_count = 0;
    double pitch_x_m = 0.0, pitch_y_m = 0.0;
    double opening_x_m = 0.0, opening_y_m = 0.0;
};

inline AperturePixelTest build_aperture_pixel_test(
    const ApertureMaterial& material) noexcept
{
    AperturePixelTest test;
    test.cos_rot = std::cos(material.rotation_rad);
    test.sin_rot = std::sin(material.rotation_rad);
    test.pattern = material.pattern;
    test.pitch_x_m = material.pitch_x_m;
    test.pitch_y_m = material.pitch_y_m;
    test.opening_x_m = material.opening_x_m;
    test.opening_y_m = material.opening_y_m;
    if (material.pattern == AperturePattern::IrisPolygon
        && material.element_count >= 3 && material.opening_x_m > 0.0
        && material.element_count <= AperturePixelTest::kMaxEdges) {
        const double radius = material.opening_x_m;
        const int n = material.element_count;
        test.edge_count = n;
        double prev_x = radius * std::cos(0.0);
        double prev_y = radius * std::sin(0.0);
        for (int edge = 0; edge < n; ++edge) {
            const double a1 =
                6.2831853071795864769 * (edge + 1) / n;
            const double x1 = radius * std::cos(a1);
            const double y1 = radius * std::sin(a1);
            test.edge_x0[static_cast<std::size_t>(edge)] = prev_x;
            test.edge_y0[static_cast<std::size_t>(edge)] = prev_y;
            test.edge_x1[static_cast<std::size_t>(edge)] = x1;
            test.edge_y1[static_cast<std::size_t>(edge)] = y1;
            prev_x = x1;
            prev_y = y1;
        }
    }
    return test;
}

inline bool aperture_open_fast(
    const AperturePixelTest& test, double x, double y) noexcept
{
    const double u = test.cos_rot*x + test.sin_rot*y;
    const double v = -test.sin_rot*x + test.cos_rot*y;
    switch (test.pattern) {
        case AperturePattern::IrisPolygon: {
            if (test.edge_count < 3) return false;
            for (int edge = 0; edge < test.edge_count; ++edge) {
                const std::size_t e = static_cast<std::size_t>(edge);
                if ((test.edge_x1[e]-test.edge_x0[e])*(v-test.edge_y0[e])
                    - (test.edge_y1[e]-test.edge_y0[e])*(u-test.edge_x0[e])
                    < 0.0)
                    return false;
            }
            return true;
        }
        case AperturePattern::ShadowMask: {
            if (test.pitch_x_m <= 0.0 || test.pitch_y_m <= 0.0
                || test.opening_x_m <= 0.0 || test.opening_y_m <= 0.0)
                return false;
            const double cell_x =
                u - std::round(u/test.pitch_x_m)*test.pitch_x_m;
            const double cell_y =
                v - std::round(v/test.pitch_y_m)*test.pitch_y_m;
            const double ex = cell_x/test.opening_x_m;
            const double ey = cell_y/test.opening_y_m;
            return ex*ex + ey*ey <= 1.0;
        }
        case AperturePattern::SlotMask: {
            if (test.pitch_x_m <= 0.0 || test.pitch_y_m <= 0.0)
                return false;
            const double cell_x =
                u - std::round(u/test.pitch_x_m)*test.pitch_x_m;
            const double cell_y =
                v - std::round(v/test.pitch_y_m)*test.pitch_y_m;
            return std::abs(cell_x) <= test.opening_x_m
                && std::abs(cell_y) <= test.opening_y_m;
        }
        case AperturePattern::ApertureGrille: {
            if (test.pitch_x_m <= 0.0) return false;
            const double cell_x =
                u - std::round(u/test.pitch_x_m)*test.pitch_x_m;
            return std::abs(cell_x) <= test.opening_x_m;
        }
        case AperturePattern::CircularHole:
            return test.opening_x_m > 0.0
                && u*u + v*v <= test.opening_x_m*test.opening_x_m;
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
    const AperturePixelTest pixel_test = build_aperture_pixel_test(material);
    double removed = 0.0;
    for (int y = 0; y < ny; ++y) {
        const double ym = y*dx - origin_y;
        for (int x = 0; x < nx; ++x) {
            const double xm = x*dx - origin_x;
            if (xm*xm + ym*ym > assembly_r2
                || aperture_open_fast(pixel_test, xm, ym))
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

bool angular_spectrum_step_wide(int bands,
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
    return angular_spectrum_wide(
        bands, nx, ny, dx, dz, wavelengths_m, direction_sign, plan, re, im);
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

template<int Bands>
static bool rigid_interface_exact(
    int nx,
    int ny,
    RigidFieldMap coordinate_map,
    const float* jones_re,
    const float* jones_im,
    const float* in_s_re,
    const float* in_s_im,
    const float* in_p_re,
    const float* in_p_im,
    float* out_s_re,
    float* out_s_im,
    float* out_p_re,
    float* out_p_im) noexcept
{
    if (nx <= 0 || ny <= 0 || !jones_re || !jones_im
        || !in_s_re || !in_s_im || !in_p_re || !in_p_im
        || !out_s_re || !out_s_im || !out_p_re || !out_p_im)
        return false;
    const int map = static_cast<int>(coordinate_map);
    if (map < 0 || map > 7 || ((map & 4) && nx != ny))
        return false;
    const bool transpose = (map & 4) != 0;
    const bool flip_x = (map & 1) != 0;
    const bool flip_y = (map & 2) != 0;
    const std::size_t plane =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    for (int band = 0; band < Bands; ++band) {
        const std::size_t band_base = static_cast<std::size_t>(band) * plane;
        const std::size_t matrix_base = static_cast<std::size_t>(band) * 4u;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                int sx = transpose ? y : x;
                int sy = transpose ? x : y;
                if (flip_x) sx = nx - 1 - sx;
                if (flip_y) sy = ny - 1 - sy;
                const std::size_t source =
                    band_base + static_cast<std::size_t>(sy) * nx + sx;
                const std::size_t destination =
                    band_base + static_cast<std::size_t>(y) * nx + x;
                const float sr = in_s_re[source];
                const float si = in_s_im[source];
                const float pr = in_p_re[source];
                const float pi = in_p_im[source];
                for (int output = 0; output < 2; ++output) {
                    const std::size_t j0 =
                        matrix_base + static_cast<std::size_t>(output) * 2u;
                    const float a_re = jones_re[j0];
                    const float a_im = jones_im[j0];
                    const float b_re = jones_re[j0 + 1u];
                    const float b_im = jones_im[j0 + 1u];
                    const float value_re =
                        a_re*sr - a_im*si + b_re*pr - b_im*pi;
                    const float value_im =
                        a_re*si + a_im*sr + b_re*pi + b_im*pr;
                    if (output == 0) {
                        out_s_re[destination] = value_re;
                        out_s_im[destination] = value_im;
                    } else {
                        out_p_re[destination] = value_re;
                        out_p_im[destination] = value_im;
                    }
                }
            }
        }
    }
    return true;
}

bool apply_rigid_field_interface(
    int bands,
    int nx,
    int ny,
    RigidFieldMap coordinate_map,
    const float* jones_re,
    const float* jones_im,
    const float* in_s_re,
    const float* in_s_im,
    const float* in_p_re,
    const float* in_p_im,
    float* out_s_re,
    float* out_s_im,
    float* out_p_re,
    float* out_p_im) noexcept
{
#define WAVE_INTERFACE_CASE(B) case B: return rigid_interface_exact<B>( \
    nx, ny, coordinate_map, jones_re, jones_im, \
    in_s_re, in_s_im, in_p_re, in_p_im, \
    out_s_re, out_s_im, out_p_re, out_p_im)
    switch (bands) {
        WAVE_INTERFACE_CASE(1);
        WAVE_INTERFACE_CASE(3);
        WAVE_INTERFACE_CASE(4);
        WAVE_INTERFACE_CASE(8);
        WAVE_INTERFACE_CASE(16);
        WAVE_INTERFACE_CASE(32);
        default: return false;
    }
#undef WAVE_INTERFACE_CASE
}

}  // namespace wave_t4
