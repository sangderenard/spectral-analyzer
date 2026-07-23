#include "lens_optics.h"
#include "thread_pool.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using V3 = Eigen::Vector3d;
using M4 = Eigen::Matrix4d;

constexpr double EPS = 1.0e-12;
constexpr int PLENS_HEADER = 8;
constexpr int PLENS_STRIDE = 8;

struct Surface {
    double x = 0.0;
    double R = 0.0;
    double n_before = 1.0;
    double n_after = 1.0;
    double aperture = 0.0;
    double k = 0.0;
    bool is_stop = false;
};

inline Surface surface_at(
    const float* p, int payload_len, int i, int spectral_lane = -1)
{
    const int off = PLENS_HEADER + i * PLENS_STRIDE;
    Surface s;
    s.x = static_cast<double>(p[off + 0]);
    s.R = static_cast<double>(p[off + 1]);
    s.n_before = static_cast<double>(p[off + 2]);
    s.n_after = static_cast<double>(p[off + 3]);
    s.aperture = static_cast<double>(p[off + 4]);
    s.k = static_cast<double>(p[off + 5]);
    s.is_stop = static_cast<double>(p[off + 6]) != 0.0;
    const int n_spectral = static_cast<int>(p[5]);
    const int spectral_offset = static_cast<int>(p[6]);
    const int spectral_stride = static_cast<int>(p[7]);
    if (!s.is_stop && spectral_lane >= 0 && spectral_lane < n_spectral
        && spectral_stride >= 2*n_spectral) {
        const int base = spectral_offset + i*spectral_stride;
        if (base >= 0 && base + n_spectral + spectral_lane < payload_len) {
            s.n_before = static_cast<double>(p[base + spectral_lane]);
            s.n_after =
                static_cast<double>(p[base + n_spectral + spectral_lane]);
        }
    }
    return s;
}

inline double conic_intersect(const V3& o, const V3& d, const Surface& s)
{
    if (std::abs(s.R) < EPS) {
        if (std::abs(d.x()) < EPS) return std::numeric_limits<double>::quiet_NaN();
        const double t = (s.x - o.x()) / d.x();
        return (t > EPS) ? t : std::numeric_limits<double>::quiet_NaN();
    }
    const double c = 1.0 / s.R;
    const double kp = 1.0 + s.k;
    const double ox = o.x() - s.x;
    const double oy = o.y();
    const double oz = o.z();
    const double dx = d.x();
    const double dy = d.y();
    const double dz = d.z();
    const double A = c * (dy * dy + dz * dz + kp * dx * dx);
    const double B = 2.0 * (c * (oy * dy + oz * dz + kp * ox * dx) - dx);
    const double C = c * (oy * oy + oz * oz + kp * ox * ox) - 2.0 * ox;
    if (std::abs(A) < EPS) {
        if (std::abs(B) < EPS) return std::numeric_limits<double>::quiet_NaN();
        const double t = -C / B;
        return (t > EPS) ? t : std::numeric_limits<double>::quiet_NaN();
    }
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double sq = std::sqrt(disc);
    const double t1 = (-B - sq) / (2.0 * A);
    const double t2 = (-B + sq) / (2.0 * A);
    const double p1x = o.x() + t1 * dx;
    const double p2x = o.x() + t2 * dx;
    if (t1 > EPS && t2 > EPS) return (std::abs(p1x - s.x) <= std::abs(p2x - s.x)) ? t1 : t2;
    if (t1 > EPS) return t1;
    if (t2 > EPS) return t2;
    return std::numeric_limits<double>::quiet_NaN();
}

inline V3 conic_normal(const V3& p, const Surface& s)
{
    if (std::abs(s.R) < EPS) return V3(1.0, 0.0, 0.0);
    const double c = 1.0 / s.R;
    const double kp = 1.0 + s.k;
    const double dx = p.x() - s.x;
    V3 n(-1.0 + kp * c * dx, c * p.y(), c * p.z());
    const double l = n.norm();
    return (l > EPS) ? n / l : V3(1.0, 0.0, 0.0);
}

inline bool snell(const V3& ray, V3 normal, double n1, double n2, V3& out)
{
    double cos_i = -ray.dot(normal);
    if (cos_i < 0.0) {
        normal = -normal;
        cos_i = -cos_i;
    }
    const double eta = n1 / n2;
    const double sin2_t = eta * eta * std::max(0.0, 1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) return false;
    const double cos_t = std::sqrt(std::max(0.0, 1.0 - sin2_t));
    out = eta * ray + (eta * cos_i - cos_t) * normal;
    const double l = out.norm();
    if (l <= EPS) return false;
    out /= l;
    return true;
}

inline bool trace_payload(
    const float* payload,
    int payload_len,
    const V3& origin,
    const V3& direction,
    V3& out_origin,
    V3& out_dir,
    int spectral_lane = -1)
{
    if (!payload || payload_len < PLENS_HEADER) return false;
    const int n = static_cast<int>(payload[1]);
    if (n <= 0 || payload_len < PLENS_HEADER + n * PLENS_STRIDE) return false;
    V3 o = origin;
    V3 d = direction;
    const double dl = d.norm();
    if (dl <= EPS) return false;
    d /= dl;

    const bool forward = d.x() >= 0.0;
    for (int step = 0; step < n; ++step) {
        const int si = forward ? step : (n - 1 - step);
        Surface s = surface_at(payload, payload_len, si, spectral_lane);
        if (!forward && !s.is_stop) std::swap(s.n_before, s.n_after);
        if (std::abs(d.x()) < EPS) return false;
        const double t = s.is_stop || std::abs(s.R) < EPS
            ? (s.x - o.x()) / d.x()
            : conic_intersect(o, d, s);
        if (!(t > EPS) || !std::isfinite(t)) return false;
        const V3 hit = o + t * d;
        if (s.aperture > 0.0 && std::hypot(hit.y(), hit.z()) > s.aperture) return false;
        V3 d2 = d;
        if (!s.is_stop) {
            const V3 normal = (std::abs(s.R) < EPS)
                ? ((d.x() < 0.0) ? V3(-1.0, 0.0, 0.0) : V3(1.0, 0.0, 0.0))
                : conic_normal(hit, s);
            if (!snell(d, normal, s.n_before, s.n_after, d2)) return false;
        }
        o = hit + d2 * (10.0 * EPS);
        d = d2;
    }
    out_origin = o;
    out_dir = d;
    return true;
}

inline V3 load3(const double* p, int i)
{
    return V3(p[3 * i + 0], p[3 * i + 1], p[3 * i + 2]);
}

} // namespace

int lens_optics_estimate_camera_jacobians(
    const float* payload,
    int payload_len,
    const double* film_origins,
    const double* aperture_points,
    const double* base_exit_origins,
    const double* base_exit_dirs,
    int n,
    double plate_radius,
    double aperture_radius,
    const double* tb,
    const double* tc,
    int n_threads,
    float* aperture_jac_out,
    float* phase_jac_out)
{
    if (!payload || !film_origins || !aperture_points || !base_exit_origins ||
        !base_exit_dirs || !tb || !tc || !aperture_jac_out || !phase_jac_out || n < 0)
        return -1;
    const V3 tbv(tb[0], tb[1], tb[2]);
    const V3 tcv(tc[0], tc[1], tc[2]);
    const V3 ey(0.0, 1.0, 0.0);
    const V3 ez(0.0, 0.0, 1.0);
    const double eps_ap = std::max(1.0e-7, std::min(1.0e-4, std::abs(aperture_radius) * 1.0e-4));
    const double eps_film = std::max(1.0e-7, std::min(1.0e-4, std::abs(plate_radius) * 1.0e-4));

    auto body = [&](size_t ii) {
        const int i = static_cast<int>(ii);
        aperture_jac_out[i] = 0.0f;
        phase_jac_out[i] = 0.0f;
        const V3 ori = load3(film_origins, i);
        const V3 ap = load3(aperture_points, i);
        const V3 base_o = load3(base_exit_origins, i);
        const V3 base_d = load3(base_exit_dirs, i);
        auto trace_variant = [&](const V3& o2, const V3& ap2, V3& oo, V3& dd) -> bool {
            V3 dir = ap2 - o2;
            const double l = dir.norm();
            if (l <= EPS) return false;
            dir /= l;
            return trace_payload(payload, payload_len, o2, dir, oo, dd);
        };

        V3 o_ay, d_ay, o_az, d_az;
        const bool ok_ay = trace_variant(ori, ap + eps_ap * tbv, o_ay, d_ay);
        const bool ok_az = trace_variant(ori, ap + eps_ap * tcv, o_az, d_az);
        if (ok_ay && ok_az) {
            const V3 ddu = (d_ay - base_d) / eps_ap;
            const V3 ddv = (d_az - base_d) / eps_ap;
            const double j = ddu.cross(ddv).norm();
            if (std::isfinite(j) && j > 0.0) aperture_jac_out[i] = static_cast<float>(j);
        }

        V3 o_fy, d_fy, o_fz, d_fz;
        const bool ok_fy = trace_variant(ori + eps_film * ey, ap, o_fy, d_fy);
        const bool ok_fz = trace_variant(ori + eps_film * ez, ap, o_fz, d_fz);
        if (ok_fy && ok_fz && ok_ay && ok_az) {
            M4 m;
            m.col(0) << (o_fy.y() - base_o.y()) / eps_film,
                        (o_fy.z() - base_o.z()) / eps_film,
                        (d_fy.y() - base_d.y()) / eps_film,
                        (d_fy.z() - base_d.z()) / eps_film;
            m.col(1) << (o_fz.y() - base_o.y()) / eps_film,
                        (o_fz.z() - base_o.z()) / eps_film,
                        (d_fz.y() - base_d.y()) / eps_film,
                        (d_fz.z() - base_d.z()) / eps_film;
            m.col(2) << (o_ay.y() - base_o.y()) / eps_ap,
                        (o_ay.z() - base_o.z()) / eps_ap,
                        (d_ay.y() - base_d.y()) / eps_ap,
                        (d_ay.z() - base_d.z()) / eps_ap;
            m.col(3) << (o_az.y() - base_o.y()) / eps_ap,
                        (o_az.z() - base_o.z()) / eps_ap,
                        (d_az.y() - base_d.y()) / eps_ap,
                        (d_az.z() - base_d.z()) / eps_ap;
            const double det = std::abs(m.determinant());
            if (std::isfinite(det) && det > 0.0) phase_jac_out[i] = static_cast<float>(det);
        }
    };

    if (n == 0) return 0;
    ThreadPool pool(static_cast<size_t>(n_threads > 0 ? n_threads : 0));
    ThreadPool::parallel_for(pool, size_t(0), static_cast<size_t>(n), body);
    return 0;
}

int lens_optics_estimate_phase_space_jacobians(
    const float* payload,
    int payload_len,
    const double* origins,
    const double* directions,
    int n,
    int spectral_lane,
    double n_input,
    double n_output,
    double q_step_m,
    double p_step,
    int n_threads,
    double* matrices_out,
    double* determinants_out,
    double* symplectic_residuals_out,
    unsigned char* valid_out)
{
    if (!payload || !origins || !directions || n < 0
        || !matrices_out || !determinants_out
        || !symplectic_residuals_out || !valid_out
        || !(n_input > 0.0) || !(n_output > 0.0)
        || !(q_step_m > 0.0) || !(p_step > 0.0))
        return -1;

    M4 omega = M4::Zero();
    omega.block<2,2>(0,2) = Eigen::Matrix2d::Identity();
    omega.block<2,2>(2,0) = -Eigen::Matrix2d::Identity();

    auto body = [&](size_t ii) {
        const int i = static_cast<int>(ii);
        double* out = matrices_out + static_cast<size_t>(i)*16u;
        std::fill(out, out + 16, 0.0);
        determinants_out[i] = 0.0;
        symplectic_residuals_out[i] =
            std::numeric_limits<double>::infinity();
        valid_out[i] = 0u;

        const V3 base_origin = load3(origins, i);
        V3 base_direction = load3(directions, i);
        const double direction_norm = base_direction.norm();
        if (!(direction_norm > EPS)) return;
        base_direction /= direction_norm;
        const double axial_sign = base_direction.x() >= 0.0 ? 1.0 : -1.0;
        const double base_state[4] = {
            base_origin.y(), base_origin.z(),
            n_input*base_direction.y(), n_input*base_direction.z()
        };

        auto trace_state = [&](const double state[4], double output[4]) -> bool {
            const double py = state[2], pz = state[3];
            const double transverse2 = (py*py + pz*pz)/(n_input*n_input);
            if (!(transverse2 < 1.0)) return false;
            V3 origin(base_origin.x(), state[0], state[1]);
            V3 direction(
                axial_sign*std::sqrt(std::max(0.0, 1.0-transverse2)),
                py/n_input, pz/n_input);
            V3 exit_origin, exit_direction;
            if (!trace_payload(
                    payload, payload_len, origin, direction,
                    exit_origin, exit_direction, spectral_lane))
                return false;
            output[0] = exit_origin.y();
            output[1] = exit_origin.z();
            output[2] = n_output*exit_direction.y();
            output[3] = n_output*exit_direction.z();
            return true;
        };

        M4 jacobian;
        for (int column = 0; column < 4; ++column) {
            double plus_state[4], minus_state[4];
            std::copy(base_state, base_state + 4, plus_state);
            std::copy(base_state, base_state + 4, minus_state);
            const double step = column < 2 ? q_step_m : p_step;
            plus_state[column] += step;
            minus_state[column] -= step;
            double plus[4], minus[4];
            if (!trace_state(plus_state, plus)
                || !trace_state(minus_state, minus))
                return;
            for (int row = 0; row < 4; ++row)
                jacobian(row, column) = (plus[row]-minus[row])/(2.0*step);
        }
        if (!jacobian.allFinite()) return;
        const double determinant = jacobian.determinant();
        const double residual =
            (jacobian.transpose()*omega*jacobian-omega).norm();
        if (!std::isfinite(determinant) || !std::isfinite(residual)) return;
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                out[row*4 + column] = jacobian(row, column);
        determinants_out[i] = determinant;
        symplectic_residuals_out[i] = residual;
        valid_out[i] = 1u;
    };

    if (n == 0) return 0;
    ThreadPool pool(static_cast<size_t>(n_threads > 0 ? n_threads : 0));
    ThreadPool::parallel_for(pool, size_t(0), static_cast<size_t>(n), body);
    return 0;
}
