// butterfly_radix4_dit.hpp
#pragma once
#include "butterfly_base.hpp"
#include <Eigen/Core>

namespace eigfft {
namespace butterfly {

class ButterflyRadix4DIT : public IButterfly {
public:
    void apply(std::complex<double>* a, std::complex<double>* b, int width, const std::complex<double>& w) const override {
        // Example: Radix-4 DIT butterfly (simplified, not production)
        for (int i = 0; i < width; i += 4) {
            // Load 4 elements
            auto x0 = a[i];
            auto x1 = a[i+1];
            auto x2 = a[i+2];
            auto x3 = a[i+3];
            // Twiddle factors (simplified)
            auto t0 = x0 + x2;
            auto t1 = x1 + x3;
            auto t2 = x0 - x2;
            auto t3 = x1 - x3;
            // Store results
            a[i]   = t0 + t1;
            a[i+1] = t2 + t3 * w;
            a[i+2] = t0 - t1;
            a[i+3] = t2 - t3 * w;
        }
    }
    void scatter(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* out0, std::complex<double>* out1, int width, const std::complex<double>& w) const override {
        // Example: Radix-4 scatter (simplified)
        for (int i = 0; i < width; i += 4) {
            out0[i]   = a[i]   + b[i]   * w;
            out0[i+1] = a[i+1] + b[i+1] * w;
            out0[i+2] = a[i+2] + b[i+2] * w;
            out0[i+3] = a[i+3] + b[i+3] * w;
            out1[i]   = a[i]   - b[i]   * w;
            out1[i+1] = a[i+1] - b[i+1] * w;
            out1[i+2] = a[i+2] - b[i+2] * w;
            out1[i+3] = a[i+3] - b[i+3] * w;
        }
    }
};

} // namespace butterfly
} // namespace eigfft
