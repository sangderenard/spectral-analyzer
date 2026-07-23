// butterfly_radix2_dit.hpp
#pragma once
#include "butterfly_base.hpp"
#include <Eigen/Core>

namespace eigfft {
namespace butterfly {

class ButterflyRadix2DIT : public IButterfly {
public:
    void apply(std::complex<double>* a, std::complex<double>* b, int width, const std::complex<double>& w) const override {
        int lane = 0;
        using Packet = Eigen::internal::packet_traits<std::complex<double>>::type;
        constexpr int PacketSize = Eigen::internal::packet_traits<std::complex<double>>::size;
        if constexpr (PacketSize > 1) {
            const Packet w_packet = Eigen::internal::pset1<Packet>(w);
            for (; lane + PacketSize <= width; lane += PacketSize) {
                const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
                const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
                const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
                const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
                const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
                Eigen::internal::pstoreu<std::complex<double>, Packet>(a + lane, sum);
                Eigen::internal::pstoreu<std::complex<double>, Packet>(b + lane, diff);
            }
        }
        for (; lane < width; ++lane) {
            const auto ai = a[lane];
            const auto bt = b[lane] * w;
            a[lane] = ai + bt;
            b[lane] = ai - bt;
        }
    }
    void scatter(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* out0, std::complex<double>* out1, int width, const std::complex<double>& w) const override {
        int lane = 0;
        using Packet = Eigen::internal::packet_traits<std::complex<double>>::type;
        constexpr int PacketSize = Eigen::internal::packet_traits<std::complex<double>>::size;
        if constexpr (PacketSize > 1) {
            const Packet w_packet = Eigen::internal::pset1<Packet>(w);
            for (; lane + PacketSize <= width; lane += PacketSize) {
                const Packet a_pack = Eigen::internal::ploadu<Packet>(a + lane);
                const Packet b_pack = Eigen::internal::ploadu<Packet>(b + lane);
                const Packet b_twiddled = Eigen::internal::pmul(b_pack, w_packet);
                const Packet sum = Eigen::internal::padd(a_pack, b_twiddled);
                const Packet diff = Eigen::internal::psub(a_pack, b_twiddled);
                Eigen::internal::pstoreu<std::complex<double>, Packet>(out0 + lane, sum);
                Eigen::internal::pstoreu<std::complex<double>, Packet>(out1 + lane, diff);
            }
        }
        for (; lane < width; ++lane) {
            out0[lane] = a[lane] + b[lane] * w;
            out1[lane] = a[lane] - b[lane] * w;
        }
    }
};

} // namespace butterfly
} // namespace eigfft
