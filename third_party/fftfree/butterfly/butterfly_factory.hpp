// butterfly_factory.hpp
#pragma once
#include "butterfly_base.hpp"
#include "butterfly_radix2_dit.hpp"
#include "butterfly_radix2_dif.hpp"

namespace eigfft {
namespace butterfly {

class ButterflyFactory {
public:
    static ButterflyPtr create(const ButterflyConfig& cfg) {
        RadixType chosen_radix = cfg.radix;
        if (!cfg.allowed_radices.empty()) {
            // Prefer allowed radix if present
            for (auto r : cfg.allowed_radices) {
                if (r == cfg.radix) { chosen_radix = r; break; }
            }
        }
        if (cfg.radix_callback) {
            chosen_radix = cfg.radix_callback(cfg);
        }
        // Select by radix, operation, flow, and simd_width
        if (chosen_radix == RadixType::Radix2 && cfg.op == OperationType::DIT) {
            return std::make_unique<ButterflyRadix2DIT>();
        } else if (chosen_radix == RadixType::Radix2 && cfg.op == OperationType::DIF) {
            return std::make_unique<ButterflyRadix2DIF>();
        } else if (chosen_radix == RadixType::Radix4 && cfg.op == OperationType::DIT) {
            return std::make_unique<ButterflyRadix4DIT>();
        }
        // Add more cases for other radix/operation/flow types as needed
        return nullptr;
    }
};

} // namespace butterfly
} // namespace eigfft
