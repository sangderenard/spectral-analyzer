// butterfly_base.hpp
// Base types and interfaces for butterfly algorithms
#pragma once
#include <complex>
#include <memory>

namespace eigfft {
namespace butterfly {

// Enum for radix type
enum class RadixType { Radix2, Radix4 };
// Enum for operation type
enum class OperationType { DIT, DIF };
// Enum for data flow type
enum class DataFlowType { InPlace, Scatter };

// Butterfly configuration
struct ButterflyConfig {
    RadixType radix;
    OperationType op;
    DataFlowType flow;
    int simd_width = 1;
    // Optional: set of allowed radices or callback for custom selection
    std::vector<RadixType> allowed_radices;
    std::function<RadixType(const ButterflyConfig&)> radix_callback;
};

// Abstract base class for butterfly algorithms
class IButterfly {
public:
    virtual ~IButterfly() = default;
    virtual void apply(std::complex<double>* a, std::complex<double>* b, int width, const std::complex<double>& w) const = 0;
    virtual void scatter(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* out0, std::complex<double>* out1, int width, const std::complex<double>& w) const = 0;
};

using ButterflyPtr = std::unique_ptr<IButterfly>;

} // namespace butterfly
} // namespace eigfft
