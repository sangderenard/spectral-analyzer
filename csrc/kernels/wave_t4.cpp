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

struct ComplexPair {
    double r;
    double i;
};

inline ComplexPair c_add(ComplexPair a, ComplexPair b) noexcept
{ return {a.r + b.r, a.i + b.i}; }
inline ComplexPair c_sub(ComplexPair a, ComplexPair b) noexcept
{ return {a.r - b.r, a.i - b.i}; }
inline ComplexPair c_mul(ComplexPair a, ComplexPair b) noexcept
{ return {a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r}; }
inline ComplexPair c_scale(ComplexPair a, double s) noexcept
{ return {a.r*s, a.i*s}; }
inline ComplexPair c_div(ComplexPair a, ComplexPair b) noexcept
{
    const double d = b.r*b.r + b.i*b.i;
    return d > 0.0
        ? ComplexPair{(a.r*b.r + a.i*b.i)/d, (a.i*b.r - a.r*b.i)/d}
        : ComplexPair{0.0, 0.0};
}

template <int B>
bool adi_exact(int nx, int ny, float dx, float dz,
               const double* wavelengths_m,
               float* re, float* im, float* tmp_re, float* tmp_im,
               float* rhs_re, float* rhs_im,
               float* cp_re, float* cp_im,
               float* dp_re, float* dp_im) noexcept
{
    if (!wavelengths_m || !re || !im || !tmp_re || !tmp_im
        || !rhs_re || !rhs_im || !cp_re || !cp_im || !dp_re || !dp_im
        || nx < 2 || ny < 2 || !(dx > 0.0f) || dz == 0.0f)
        return false;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const double dx2 = static_cast<double>(dx) * dx;

    auto rhs_at = [&](int i) -> ComplexPair { return {rhs_re[i], rhs_im[i]}; };
    auto cp_at = [&](int i) -> ComplexPair { return {cp_re[i], cp_im[i]}; };
    auto dp_at = [&](int i) -> ComplexPair { return {dp_re[i], dp_im[i]}; };
    auto store_rhs = [&](int i, ComplexPair v) { rhs_re[i]=(float)v.r; rhs_im[i]=(float)v.i; };
    auto store_cp = [&](int i, ComplexPair v) { cp_re[i]=(float)v.r; cp_im[i]=(float)v.i; };
    auto store_dp = [&](int i, ComplexPair v) { dp_re[i]=(float)v.r; dp_im[i]=(float)v.i; };

    auto thomas = [&](int n, ComplexPair beta) {
        const ComplexPair diag{1.0 + 2.0*beta.r, 2.0*beta.i};
        const ComplexPair off{-beta.r, -beta.i};
        store_rhs(0, {0.0, 0.0});
        store_rhs(n - 1, {0.0, 0.0});
        store_cp(0, c_div(off, diag));
        store_dp(0, c_div(rhs_at(0), diag));
        for (int i = 1; i < n; ++i) {
            const ComplexPair denom = c_sub(diag, c_mul(off, cp_at(i - 1)));
            store_cp(i, c_div(off, denom));
            store_dp(i, c_div(c_sub(rhs_at(i), c_mul(off, dp_at(i - 1))), denom));
        }
        store_rhs(n - 1, dp_at(n - 1));
        for (int i = n - 2; i >= 0; --i)
            store_rhs(i, c_sub(dp_at(i), c_mul(cp_at(i), rhs_at(i + 1))));
    };

    for (int band = 0; band < B; ++band) {
        if (!(wavelengths_m[band] > 0.0)) continue;
        const std::size_t boff = static_cast<std::size_t>(band) * plane;
        const double k = 6.283185307179586476925286766559 / wavelengths_m[band];
        const double phase = k * dz;
        const double pc = std::cos(phase), ps = std::sin(phase);
        for (std::size_t p = 0; p < plane; ++p) {
            const std::size_t i = boff + p;
            const double ar = re[i], ai = im[i];
            re[i] = static_cast<float>(ar*pc - ai*ps);
            im[i] = static_cast<float>(ar*ps + ai*pc);
        }
        const ComplexPair beta{0.0, static_cast<double>(dz) / (4.0*k*dx2)};
        for (int y = 0; y < ny; ++y) {
            const std::size_t row = boff + static_cast<std::size_t>(y) * nx;
            for (int x = 0; x < nx; ++x) {
                const ComplexPair u{re[row+x], im[row+x]};
                const ComplexPair w = x > 0
                    ? ComplexPair{re[row+x-1], im[row+x-1]} : ComplexPair{0.0,0.0};
                const ComplexPair e = x+1 < nx
                    ? ComplexPair{re[row+x+1], im[row+x+1]} : ComplexPair{0.0,0.0};
                store_rhs(x, c_add(u, c_mul(beta, c_sub(c_add(w,e), c_scale(u,2.0)))));
            }
            thomas(nx, beta);
            for (int x = 0; x < nx; ++x) {
                const ComplexPair v = rhs_at(x);
                tmp_re[row+x]=(float)v.r; tmp_im[row+x]=(float)v.i;
            }
        }
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                const std::size_t i = boff + static_cast<std::size_t>(y)*nx + x;
                const ComplexPair u{tmp_re[i], tmp_im[i]};
                const ComplexPair n = y > 0
                    ? ComplexPair{tmp_re[i-nx], tmp_im[i-nx]} : ComplexPair{0.0,0.0};
                const ComplexPair s = y+1 < ny
                    ? ComplexPair{tmp_re[i+nx], tmp_im[i+nx]} : ComplexPair{0.0,0.0};
                store_rhs(y, c_add(u, c_mul(beta, c_sub(c_add(n,s), c_scale(u,2.0)))));
            }
            thomas(ny, beta);
            for (int y = 0; y < ny; ++y) {
                const std::size_t i = boff + static_cast<std::size_t>(y)*nx + x;
                const ComplexPair v = rhs_at(y);
                re[i]=(float)v.r; im[i]=(float)v.i;
            }
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

}  // namespace

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
                     float* dp_im) noexcept
{
#define WAVE_ADI_CASE(B) case B: return adi_exact<B>(nx, ny, dx, dz, wavelengths_m, \
    re, im, tmp_re, tmp_im, rhs_re, rhs_im, cp_re, cp_im, dp_re, dp_im)
    switch (bands) {
        WAVE_ADI_CASE(1);
        WAVE_ADI_CASE(3);
        WAVE_ADI_CASE(4);
        WAVE_ADI_CASE(8);
        WAVE_ADI_CASE(16);
        WAVE_ADI_CASE(32);
        default: return false;
    }
#undef WAVE_ADI_CASE
}

}  // namespace wave_t4
