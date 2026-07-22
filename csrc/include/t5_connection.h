/**
 * t5_connection.h  —  BDPT T5 connection-pass helpers.
 *
 * INTERNAL HEADER.  Include only from ray_tracer.cpp, after these items are
 * already present in the translation unit:
 *
 *   Types    : RayTracerState, BVHNode, Triangle, MatSpectralCache,
 *              RtMaterialSurfaceCache
 *   Constants: T_SELF
 *   Functions: bvh_query(), mat_cache_diffusion(), surf_cache_ggx_alpha(),
 *              ggx_pdf_solid_angle(), band_to_display_rgb()
 *
 * Exposes:
 *   BdptSubpathView       — sorted, deduplicated per-subpath pointer view
 *   LightVertRef          — light vertex back-reference
 *   BdptCandidateScratch  — reusable per-chain MIS scratch storage
 *   T5ThreadAccum         — per-worker pixel + connection accumulator
 *   T5ConnContext         — all per-pass state + connection helper methods
 *   T5GpuParams           — GPU params block for t5_full_connect.comp.glsl
 *   run_t5_allpairs()     — full brute-force ThreadPool dispatch (CPU fallback)
 */
#pragma once

#include "ray_pipeline.h"   /* RayPipelineState, GlPipelineDispatch */
#include "thread_pool.h"    /* ThreadPool */

/* T5_LGV_STRIDE, T5_CGV_STRIDE, T5GpuParams — defined in ray_pipeline.h */

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

/* ─────────────────────────────────────────────────────────────────────────
 * BdptSubpathView
 *
 * Pointer view of one subpath's vertices sorted by vertex_index, plus the
 * cumulative forward-PDF prefix needed for MIS weight evaluation.
 * ───────────────────────────────────────────────────────────────────────── */
struct BdptSubpathView {
    std::vector<const BdptVertexRecord*> v;
    std::vector<double>  prefix_pdf;
    std::vector<uint8_t> prefix_valid;
};

/* ─────────────────────────────────────────────────────────────────────────
 * LightVertRef
 *
 * Back-reference from a light vert flat-array slot to its logical subpath and
 * vertex index.
 * ───────────────────────────────────────────────────────────────────────── */
struct LightVertRef {
    const BdptVertexRecord* rec;
    const BdptSubpathView*  sp;
    uint32_t                li;   /* index into sp->v */
};

/* ─────────────────────────────────────────────────────────────────────────
 * BdptCandidateScratch
 *
 * Reusable temporary storage for one candidate (s,t) strategy chain during
 * MIS density evaluation.  Allocate once per worker thread, reuse each call.
 * ───────────────────────────────────────────────────────────────────────── */
struct BdptCandidateScratch {
    std::vector<const BdptVertexRecord*> chain;
    std::vector<double>  edge_fwd,    edge_bwd;
    std::vector<uint8_t> edge_fwd_ok, edge_bwd_ok;
    std::vector<double>  prefix_fwd,  suffix_bwd;
    std::vector<uint8_t> prefix_ok,   suffix_ok;

    BdptCandidateScratch() {
        chain.reserve(32);
        edge_fwd.reserve(31);    edge_bwd.reserve(31);
        edge_fwd_ok.reserve(31); edge_bwd_ok.reserve(31);
        prefix_fwd.reserve(32);  suffix_bwd.reserve(32);
        prefix_ok.reserve(32);   suffix_ok.reserve(32);
    }
};

/* ─────────────────────────────────────────────────────────────────────────
 * T5ThreadAccum
 *
 * Per-worker-thread accumulation buffers.  One instance per thread; merged
 * into ps->sensor_accum (under sensor_mu) after all workers have finished.
 * ───────────────────────────────────────────────────────────────────────── */
struct T5ThreadAccum {
    std::vector<double>              sensor_r, sensor_g, sensor_b;
    std::vector<BdptConnectionRecord> conn_batch;
    uint64_t                         exact_count = 0;
    uint64_t                         visible_pairs = 0;
    uint64_t                         overlapping_bands = 0;
    uint64_t                         camera_response_bands = 0;
    uint64_t                         light_response_bands = 0;
    uint64_t                         positive_bands = 0;

    explicit T5ThreadAccum(size_t pix)
        : sensor_r(pix, 0.0), sensor_g(pix, 0.0), sensor_b(pix, 0.0)
    {
        conn_batch.reserve(4096);
    }

    /* Non-copyable (large pixel buffers); moveable. */
    T5ThreadAccum(const T5ThreadAccum&)            = delete;
    T5ThreadAccum& operator=(const T5ThreadAccum&) = delete;
    T5ThreadAccum(T5ThreadAccum&&)                 = default;
    T5ThreadAccum& operator=(T5ThreadAccum&&)      = default;
};

/* ─────────────────────────────────────────────────────────────────────────
 * T5ConnContext
 *
 * All per-pass state needed by the connection helpers.  Construct once in
 * ray_pipeline_run_bdpt_connection(), then pass by reference to the strategy
 * dispatch functions.  The member methods are the former lambda bodies.
 * ───────────────────────────────────────────────────────────────────────── */
struct T5ConnContext {

    /* ── LUT references (non-owning; lifetime == enclosing function scope) */
    const std::unordered_map<uint64_t, std::complex<float>>&
        beta_lut;
    const std::unordered_map<uint64_t, uint32_t>& spectral_sample_lut;
    const std::unordered_map<uint64_t, std::vector<BdptPdfRecord>>&
        pdf_lut;
    const std::unordered_map<uint64_t, std::vector<BdptOpticalEventRecord>>&
        optical_lut;

    /* ── Pipeline state */
    RayPipelineState* ps;

    /* ── Scalar pass parameters */
    int    n_bands;
    int    res;
    float  inv_w;
    float  inv_h;
    float  t5_min_geom;

    /* ── Spectral frequency table for RGB display mapping */
    const Eigen::VectorXd& t5_freq_hz;

    /* ── Shared connection-record flush mutex */
    std::mutex& conn_mu;

    /* ── Heartbeat counters (written by workers; read by heartbeat thread).
     * Intentional benign data race on hb_done reads in the heartbeat —
     * approximate progress is acceptable there.                          */
    mutable std::atomic<size_t> hb_done{0};
    size_t                      hb_total{0};

    /* ── Constructor ──────────────────────────────────────────────────── */
    T5ConnContext(
        const std::unordered_map<uint64_t, std::complex<float>>&              bl,
        const std::unordered_map<uint64_t, uint32_t>&                         sl,
        const std::unordered_map<uint64_t, std::vector<BdptPdfRecord>>&       pl,
        const std::unordered_map<uint64_t, std::vector<BdptOpticalEventRecord>>& ol,
        RayPipelineState*      pipeline_state,
        int    nb,
        int    r,
        float  iw,
        float  ih,
        float  min_geom,
        const Eigen::VectorXd& fhz,
        std::mutex&            mu)
        : beta_lut(bl), spectral_sample_lut(sl), pdf_lut(pl), optical_lut(ol)
        , ps(pipeline_state)
        , n_bands(nb), res(r), inv_w(iw), inv_h(ih), t5_min_geom(min_geom)
        , t5_freq_hz(fhz)
        , conn_mu(mu)
    {}

    uint32_t spectral_sample_for(const BdptVertexRecord& v) const
    {
        const uint64_t k = ((uint64_t)v.subpath_id << 32) | v.vertex_index;
        auto it = spectral_sample_lut.find(k);
        return it == spectral_sample_lut.end() ? 0u : it->second;
    }

    bool spectral_samples_compatible(
        const BdptVertexRecord& camera, const BdptVertexRecord& light) const
    {
        const uint32_t c = spectral_sample_for(camera);
        const uint32_t l = spectral_sample_for(light);
        return (c == 0u && l == 0u) || (c != 0u && c == l);
    }

    /* ─── LUT queries ─────────────────────────────────────────────────── */

    const std::vector<BdptOpticalEventRecord>*
    optical_for(const BdptVertexRecord& v) const
    {
        const uint64_t k = ((uint64_t)v.subpath_id   << 32)
                         | ((uint64_t)v.vertex_index  << 16)
                         | (uint64_t)v.stream;
        auto it = optical_lut.find(k);
        return (it == optical_lut.end()) ? nullptr : &it->second;
    }

    const BdptPdfRecord*
    pdf_for(uint32_t sid, uint16_t vi) const
    {
        const uint64_t k = ((uint64_t)sid << 32) | (uint64_t)vi;
        auto it = pdf_lut.find(k);
        if (it == pdf_lut.end() || it->second.empty()) return nullptr;

        const uint8_t domains[] = {
            BDPT_DOMAIN_PROJ_SOLID_ANGLE,
            BDPT_DOMAIN_SOLID_ANGLE,
            BDPT_DOMAIN_AREA,
            BDPT_DOMAIN_UNKNOWN
        };
        for (uint8_t d : domains)
            for (const auto& pr : it->second)
                if (pr.sample_domain == d) return &pr;
        return &it->second.front();
    }

    double beta_scalar_val(const BdptVertexRecord& v) const
    {
        double beta = 0.0;
        bool have = false;
        for (int b = 0; b < n_bands; ++b) {
            const uint64_t k = ((uint64_t)v.subpath_id  << 32)
                             | ((uint64_t)v.vertex_index << 16)
                             | (uint64_t)b;
            auto it = beta_lut.find(k);
            if (it == beta_lut.end()) break;
            beta += std::abs(it->second);
            have = true;
        }
        return have ? beta : (double)v.throughput_scalar;
    }

    double beta_band_val(const BdptVertexRecord& v, int band) const
    {
        const uint64_t k = ((uint64_t)v.subpath_id << 32)
                         | ((uint64_t)v.vertex_index << 16)
                         | (uint64_t)band;
        auto it = beta_lut.find(k);
        return (it == beta_lut.end()) ? 0.0 : std::abs(it->second);
    }

    /* ─── Core path-transport helpers ────────────────────────────────── */

    bool optical_allows_sampling(const BdptVertexRecord& v, bool reverse,
                                  double& out_jacobian) const
    {
        out_jacobian = 1.0;
        const auto* oes = optical_for(v);
        if (oes) {
            for (const auto& oe : *oes) {
                if (oe.reason == BDPT_OPT_ABSORPTION  ||
                    oe.reason == BDPT_OPT_APERTURE_CLIP ||
                    oe.reason == BDPT_OPT_VIGNETTE_CLIP ||
                    oe.reason == BDPT_OPT_TIR)
                    return false;
                if (oe.phase_space_jacobian > 0.0f &&
                    std::isfinite(oe.phase_space_jacobian)) {
                    const double j = static_cast<double>(oe.phase_space_jacobian);
                    out_jacobian *= reverse ? (1.0 / j) : j;
                }
            }
        }
        return out_jacobian > 0.0 && std::isfinite(out_jacobian);
    }

    bool edge_pdf_area_component(const BdptVertexRecord& pdf_vertex,
                                  const BdptVertexRecord& sampler_from,
                                  const BdptVertexRecord& target,
                                  bool  reverse_pdf,
                                  double& out_pdf) const
    {
        const BdptPdfRecord* pr = pdf_for(pdf_vertex.subpath_id,
                                          pdf_vertex.vertex_index);
        if (!pr) return false;

        double optical_j = 1.0;
        if (!optical_allows_sampling(pdf_vertex, reverse_pdf, optical_j))
            return false;

        double pdf = 0.0;
        const float pdf_component = reverse_pdf ? pr->pdf_rev : pr->pdf_fwd;
        const float pdf_solid     = reverse_pdf ? pr->pdf_rev : pr->pdf_solid_angle;

        /* Delta-specular sampling (including stochastic Fresnel refraction)
         * has a discrete probability mass.  It must not acquire the
         * cos(theta)/distance^2 Jacobian used to convert continuous
         * solid-angle densities to area measure. */
        if ((pr->flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) {
            if (pdf_component > 0.0f && std::isfinite(pdf_component))
                pdf = (double)pdf_component;
        } else if (!reverse_pdf && pr->pdf_area > 0.0f && std::isfinite(pr->pdf_area)) {
            pdf = (double)pr->pdf_area;
        } else {
            const Eigen::Vector3d p0(sampler_from.pos[0],
                                     sampler_from.pos[1],
                                     sampler_from.pos[2]);
            const Eigen::Vector3d p1(target.pos[0], target.pos[1], target.pos[2]);
            Eigen::Vector3d d = p1 - p0;
            const double dist2 = d.squaredNorm();
            if (dist2 <= 1e-18) { out_pdf = 1e-12; return true; }
            d /= std::sqrt(dist2);
            const Eigen::Vector3d n1(target.normal[0],
                                     target.normal[1],
                                     target.normal[2]);
            const double cos_to = std::max(0.0, std::abs(n1.dot(-d)));
            if (cos_to <= 0.0) return false;
            if (pdf_solid > 0.0f)
                pdf = (double)pdf_solid * cos_to / dist2;
            else if (pdf_component > 0.0f)
                pdf = (double)pdf_component * cos_to / dist2;
        }

        pdf *= optical_j;
        if (!(pdf > 0.0) || !std::isfinite(pdf)) return false;
        out_pdf = std::max(pdf, 1e-12);
        return true;
    }

    bool edge_pdf_area(const BdptVertexRecord& from,
                        const BdptVertexRecord& to,
                        double& out_pdf) const
    {
        return edge_pdf_area_component(from, from, to, false, out_pdf);
    }

    bool scatter_connection_pdf_area(const BdptVertexRecord& from,
                                      const BdptVertexRecord& to,
                                      double& out_pdf) const
    {
        const BdptPdfRecord* pr = pdf_for(from.subpath_id, from.vertex_index);
        if (!pr) return false;
        if ((pr->flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return false;
        if (from.mat_idx < 0) return false;

        const double diffuse_p = static_cast<double>(
            mat_cache_diffusion(ps->st->mat_cache, from.mat_idx));
        const double spec_p = std::max(0.0, 1.0 - diffuse_p);

        double optical_j = 1.0;
        if (!optical_allows_sampling(from, false, optical_j)) return false;

        const Eigen::Vector3d p0(from.pos[0], from.pos[1], from.pos[2]);
        const Eigen::Vector3d p1(to.pos[0],   to.pos[1],   to.pos[2]);
        Eigen::Vector3d d = p1 - p0;
        const double dist2 = d.squaredNorm();
        if (dist2 <= 1e-18) { out_pdf = 1e-12; return true; }
        d /= std::sqrt(dist2);

        const Eigen::Vector3d n0(from.normal[0], from.normal[1], from.normal[2]);
        const Eigen::Vector3d n1(to.normal[0],   to.normal[1],   to.normal[2]);
        const double cos_out = std::max(0.0, n0.dot(d));
        const double cos_to  = std::max(0.0, std::abs(n1.dot(-d)));
        if (cos_out <= 0.0 || cos_to <= 0.0) return false;

        double pdf_sa = 0.0;
        if ((pr->flags & BDPT_PDF_FLAG_EMISSION) != 0u) {
            pdf_sa = cos_out / M_PI;
        } else {
            if ((pr->flags & BDPT_PDF_FLAG_DIFFUSE) != 0u && diffuse_p > 0.0)
                pdf_sa += diffuse_p * cos_out / M_PI;
            if ((pr->flags & BDPT_PDF_FLAG_GGX) != 0u && spec_p > 0.0) {
            const double alpha = static_cast<double>(
                surf_cache_ggx_alpha(ps->st->surf_cache, from.mat_idx));
                if (alpha > 1.0e-3) {
                    const Eigen::Vector3d in_dir(from.dir_in[0],
                                                 from.dir_in[1],
                                                 from.dir_in[2]);
                    pdf_sa += spec_p * ggx_pdf_solid_angle(n0, in_dir, d, alpha);
                }
            }
        }

        if (!(pdf_sa > 0.0) || !std::isfinite(pdf_sa)) return false;
        const double pdf = pdf_sa * cos_to / dist2 * optical_j;
        if (!(pdf > 0.0) || !std::isfinite(pdf)) return false;
        out_pdf = std::max(pdf, 1e-12);
        return true;
    }

    bool vertex_connectable(const BdptVertexRecord& v) const
    {
        if (v.tri_id < 0) return false;
        if ((v.flags & MAT_FLAG_APERTURE_STOP) != 0) return false;
        const auto* oes = optical_for(v);
        if (oes) {
            for (const auto& oe : *oes) {
                if (oe.reason == BDPT_OPT_ABSORPTION  ||
                    oe.reason == BDPT_OPT_APERTURE_CLIP ||
                    oe.reason == BDPT_OPT_VIGNETTE_CLIP ||
                    oe.reason == BDPT_OPT_TIR)
                    return false;
            }
        }
        return true;
    }

    double endpoint_response(const BdptVertexRecord& v, int band,
                             const Eigen::Vector3d& out_dir,
                             bool sensor_endpoint = false) const
    {
        const BdptPdfRecord* pr = pdf_for(v.subpath_id, v.vertex_index);
        if (!pr) return 0.0;
        /* The camera endpoint is a measurement response, not a surface BRDF.
         * Its directional support is represented by the launch PDF/MIS chain. */
        if (sensor_endpoint) return 1.0;
        if (v.mat_idx < 0) return 1.0;
        Eigen::Vector3d n(v.normal[0], v.normal[1], v.normal[2]);
        Eigen::Vector3d wi(v.dir_in[0], v.dir_in[1], v.dir_in[2]);
        const Eigen::Vector3d wo = out_dir.normalized();
        const double cos_out = std::max(0.0, n.dot(wo));
        if (cos_out <= 0.0) return 0.0;
        if ((pr->flags & BDPT_PDF_FLAG_DELTA_SPECULAR) != 0u) return 0.0;
        if ((pr->flags & BDPT_PDF_FLAG_EMISSION) != 0u) return 1.0;

        const double refl = std::abs(mat_cache_refl(ps->st->mat_cache, v.mat_idx, band));
        if (!(refl > 0.0)) return 0.0;
        const double dp = std::clamp(
            static_cast<double>(mat_cache_diffusion(ps->st->mat_cache, v.mat_idx)), 0.0, 1.0);
        double response = 0.0;
        if ((pr->flags & BDPT_PDF_FLAG_DIFFUSE) != 0u && dp > 0.0)
            response += refl * dp / M_PI;
        if ((pr->flags & BDPT_PDF_FLAG_GGX) != 0u) {
            const double sp = std::max(0.0, 1.0 - dp);
            const double alpha = static_cast<double>(
                surf_cache_ggx_alpha(ps->st->surf_cache, v.mat_idx));
            const Eigen::Vector3d V = (-wi).normalized();
            const double NoV = std::max(0.0, n.dot(V));
            const double NoL = std::max(0.0, n.dot(wo));
            Eigen::Vector3d H = V + wo;
            const double h2 = H.squaredNorm();
            if (sp > 0.0 && alpha > 1.0e-3 && NoV > 0.0 && NoL > 0.0 && h2 > 1.0e-20) {
                H /= std::sqrt(h2);
                const double NoH = std::max(0.0, n.dot(H));
                const double VoH = std::max(0.0, V.dot(H));
                if (NoH > 0.0 && VoH > 1.0e-12) {
                    const double D = ggx_D(alpha, NoH);
                    const double G = ggx_G1(alpha, NoV) * ggx_G1(alpha, NoL);
                    response += sp * refl * D * G / std::max(4.0 * NoV * NoL, 1.0e-12);
                }
            }
        }
        return (response > 0.0 && std::isfinite(response)) ? response : 0.0;
    }

    bool visible_with_glass(const BdptVertexRecord& a, const BdptVertexRecord& b,
                            std::vector<double>& glass_t) const
    {
        const RayTracerState& st = *ps->st;
        glass_t.assign(static_cast<size_t>(n_bands), 1.0);
        if (st.bvh_nodes.empty()) return true;
        Eigen::Vector3d p0(a.pos[0], a.pos[1], a.pos[2]);
        Eigen::Vector3d p1(b.pos[0], b.pos[1], b.pos[2]);
        Eigen::Vector3d d = p1 - p0;
        const double dist = d.norm();
        if (dist <= 1e-9) return false;
        d /= dist;
        const Eigen::Vector3d inv(1.0 / d.x(), 1.0 / d.y(), 1.0 / d.z());
        Eigen::Vector3d orig = p0 + d * (T_SELF * 16.0);
        double remaining = std::max(0.0, dist - T_SELF * 32.0);
        for (int crossing = 0; crossing < 64 && remaining > T_SELF; ++crossing) {
            double t_hit = remaining;
            int hit_tri = -1;
            bvh_query(st.bvh_nodes, st.bvh_tri_ids, st.tris,
                      orig, d, inv, t_hit, hit_tri);
            if (hit_tri < 0) return true;
            const Triangle& tri = st.tris[static_cast<size_t>(hit_tri)];
            if (hit_tri != a.tri_id && hit_tri != b.tri_id) {
                if ((tri.flags & MAT_FLAG_TRANSMISSIVE) == 0) return false;
                const bool front = d.dot(tri.normal) < 0.0;
                const int m1 = front ? tri.medium_pos_mat_idx : tri.medium_neg_mat_idx;
                const int m2 = front ? tri.medium_neg_mat_idx : tri.medium_pos_mat_idx;
                const double ci = std::clamp(std::abs(d.dot(tri.normal)), 0.0, 1.0);
                for (int band = 0; band < n_bands; ++band) {
                    const double n1 = (m1 >= 0) ? std::max(1.0e-10, (double)mat_band_record(st,m1,band)[7]) : 1.0;
                    const double n2 = (m2 >= 0) ? std::max(1.0e-10, (double)mat_band_record(st,m2,band)[7]) : 1.0;
                    const double sin2_t = (n1/n2)*(n1/n2)*(1.0-ci*ci);
                    if (sin2_t > 1.0) { glass_t[static_cast<size_t>(band)] = 0.0; continue; }
                    const double ct = std::sqrt(std::max(0.0, 1.0-sin2_t));
                    glass_t[static_cast<size_t>(band)] *=
                        std::sqrt(std::max(0.0, 1.0-fresnel_R(ci,ct,n1,n2)));
                }
            }
            const double advance = t_hit + T_SELF * 16.0;
            orig += d * advance;
            remaining -= advance;
        }
        return remaining <= T_SELF;
    }

    /* ─── MIS density ────────────────────────────────────────────────── */

    bool candidate_strategy_density(
        const BdptSubpathView& cam,   size_t ci,
        const BdptSubpathView& light, size_t li,
        BdptCandidateScratch&  scratch,
        double&                out_selected_pdf,
        double&                out_denom) const
    {
        auto& chain = scratch.chain;
        chain.clear();
        chain.reserve(ci + li + 2);
        for (size_t i = 0; i <= ci; ++i) chain.push_back(cam.v[i]);
        for (size_t n = 0; n <= li; ++n) chain.push_back(light.v[li - n]);

        const size_t N = chain.size();
        if (N < 2) return false;

        auto& edge_fwd    = scratch.edge_fwd;
        auto& edge_bwd    = scratch.edge_bwd;
        auto& edge_fwd_ok = scratch.edge_fwd_ok;
        auto& edge_bwd_ok = scratch.edge_bwd_ok;
        edge_fwd.resize(N - 1);    edge_bwd.resize(N - 1);
        edge_fwd_ok.resize(N - 1); edge_bwd_ok.resize(N - 1);
        std::fill(edge_fwd.begin(),    edge_fwd.end(),    0.0);
        std::fill(edge_bwd.begin(),    edge_bwd.end(),    0.0);
        std::fill(edge_fwd_ok.begin(), edge_fwd_ok.end(), 0u);
        std::fill(edge_bwd_ok.begin(), edge_bwd_ok.end(), 0u);

        for (size_t e = 0; e + 1 < N; ++e) {
            double pf = 0.0, pb = 0.0;
            bool ok_f = false, ok_b = false;
            if (e < ci) {
                const auto& parent = *cam.v[e];
                const auto& child  = *cam.v[e + 1];
                ok_f = edge_pdf_area_component(parent, parent, child, false, pf);
                ok_b = edge_pdf_area_component(parent, child, parent, true,  pb);
            } else if (e == ci) {
                ok_f = scatter_connection_pdf_area(*chain[e],     *chain[e + 1], pf);
                ok_b = scatter_connection_pdf_area(*chain[e + 1], *chain[e],     pb);
            } else {
                const size_t lci = li - (e - ci - 1);
                if (lci == 0) return false;
                const auto& parent = *light.v[lci - 1];
                const auto& child  = *light.v[lci];
                ok_f = edge_pdf_area_component(parent, child,  parent, true,  pf);
                ok_b = edge_pdf_area_component(parent, parent, child,  false, pb);
            }
            if (ok_f) { edge_fwd[e] = pf; edge_fwd_ok[e] = 1u; }
            if (ok_b) { edge_bwd[e] = pb; edge_bwd_ok[e] = 1u; }
        }

        auto& prefix_fwd = scratch.prefix_fwd; auto& suffix_bwd = scratch.suffix_bwd;
        auto& prefix_ok  = scratch.prefix_ok;  auto& suffix_ok  = scratch.suffix_ok;
        prefix_fwd.resize(N); suffix_bwd.resize(N);
        prefix_ok.resize(N);  suffix_ok.resize(N);
        std::fill(prefix_fwd.begin(), prefix_fwd.end(), 0.0);
        std::fill(suffix_bwd.begin(), suffix_bwd.end(), 0.0);
        std::fill(prefix_ok.begin(),  prefix_ok.end(),  0u);
        std::fill(suffix_ok.begin(),  suffix_ok.end(),  0u);

        prefix_fwd[0] = 1.0; prefix_ok[0] = 1u;
        for (size_t i = 1; i < N; ++i) {
            prefix_ok[i]  = (prefix_ok[i-1] && edge_fwd_ok[i-1]) ? 1u : 0u;
            prefix_fwd[i] = prefix_ok[i]
                ? std::max(prefix_fwd[i-1] * edge_fwd[i-1], 1e-300) : 0.0;
        }
        suffix_bwd[N-1] = 1.0; suffix_ok[N-1] = 1u;
        for (size_t i = N-1; i-- > 0;) {
            suffix_ok[i]  = (suffix_ok[i+1] && edge_bwd_ok[i]) ? 1u : 0u;
            suffix_bwd[i] = suffix_ok[i]
                ? std::max(suffix_bwd[i+1] * edge_bwd[i], 1e-300) : 0.0;
        }

        const size_t selected_cut = ci + 1;
        if (selected_cut == 0 || selected_cut >= N)          return false;
        if (!prefix_ok[selected_cut-1] || !suffix_ok[selected_cut]) return false;

        out_selected_pdf = std::max(
            prefix_fwd[selected_cut-1] * suffix_bwd[selected_cut], 1e-300);
        out_denom = 0.0;
        for (size_t cut = 1; cut < N; ++cut) {
            if (!vertex_connectable(*chain[cut-1])) continue;
            if (!vertex_connectable(*chain[cut]))   continue;
            if (!prefix_ok[cut-1] || !suffix_ok[cut]) continue;
            const double p = prefix_fwd[cut-1] * suffix_bwd[cut];
            if (p > 0.0 && std::isfinite(p)) out_denom += p;
        }
        return out_selected_pdf > 0.0 &&
               out_denom        > 0.0 &&
               std::isfinite(out_denom);
    }

    /* ─── Connection helpers ─────────────────────────────────────────── */

    /* Flush accumulated connection records under the shared mutex. */
    void flush_connections(T5ThreadAccum& acc) const
    {
        if (acc.conn_batch.empty()) return;
        std::lock_guard<std::mutex> lk(conn_mu);
        for (auto& cr : acc.conn_batch)
            ps->push_bdpt_connection(std::move(cr));
        acc.conn_batch.clear();
    }

    /* Evaluate one camera vertex vs one light vertex.
     * Appends a BdptConnectionRecord and accumulates the visible contribution
     * into pixel_accum.  Returns without side-effects if any filter fails. */
    void try_connect_pair(
        const BdptSubpathView&  cam,      size_t ci,
        const BdptVertexRecord& c,
        const BdptSubpathView&  light,    size_t li,
        BdptCandidateScratch&   scratch,
        T5ThreadAccum&          acc,
        double& out_r, double& out_g, double& out_b) const
    {
        const BdptVertexRecord& l = *light.v[li];
        if (!vertex_connectable(l))  return;
        if (!light.prefix_valid[li]) return;
        if (!spectral_samples_compatible(c, l)) return;

        const float dx    = l.pos[0] - c.pos[0];
        const float dy    = l.pos[1] - c.pos[1];
        const float dz    = l.pos[2] - c.pos[2];
        const float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 < 1e-12f) return;
        const float dist  = std::sqrt(dist2);
        const float cx = dx/dist, cy = dy/dist, cz = dz/dist;
        const float cos_c = std::abs(c.normal[0]*cx    + c.normal[1]*cy    + c.normal[2]*cz);
        const float cos_l = std::abs(l.normal[0]*(-cx) + l.normal[1]*(-cy) + l.normal[2]*(-cz));
        const float geom  = cos_c * cos_l / dist2;
        if (geom < t5_min_geom) return;

        double strategy_pdf = 0.0, denom = 0.0;
        if (!candidate_strategy_density(cam, ci, light, li, scratch,
                                         strategy_pdf, denom))
            return;
        const double mis_weight = strategy_pdf / denom;

        std::vector<double> glass_t;
        const bool is_visible = visible_with_glass(c, l, glass_t);

        BdptConnectionRecord cr{};
        cr.camera_subpath_id   = c.subpath_id;
        cr.light_subpath_id    = l.subpath_id;
        cr.camera_vertex_index = c.vertex_index;
        cr.light_vertex_index  = l.vertex_index;
        cr.strategy_s          = static_cast<uint16_t>(
            std::min<size_t>(ci + 1, 0xFFFFu));
        cr.strategy_t          = static_cast<uint16_t>(
            std::min<size_t>(li + 1, 0xFFFFu));
        cr.flags               = is_visible ? 0u : (1u << 0);
        cr.p0[0] = c.pos[0]; cr.p0[1] = c.pos[1]; cr.p0[2] = c.pos[2];
        cr.p1[0] = l.pos[0]; cr.p1[1] = l.pos[1]; cr.p1[2] = l.pos[2];
        cr.dist2               = dist2;
        cr.cos_camera          = cos_c;
        cr.cos_light           = cos_l;
        cr.geometry_term       = geom;
        cr.visibility          = is_visible ? 1.0f : 0.0f;
        cr.strategy_pdf        = static_cast<float>(
            std::min(strategy_pdf, (double)std::numeric_limits<float>::max()));
        cr.mis_weight          = static_cast<float>(mis_weight);
        acc.conn_batch.push_back(cr);
        if (acc.conn_batch.size() >= 4096) flush_connections(acc);
        if (!is_visible) return;
        ++acc.visible_pairs;

        /* Identical spectral contract to t5_full_connect: bands overlap by
         * index, both endpoint BSDFs are evaluated for the actual connection
         * direction, and transmissive crossings apply per-band Fresnel
         * amplitude.  Cross-band scalar products are forbidden. */
        const Eigen::Vector3d c_to_l(cx, cy, cz);
        const Eigen::Vector3d l_to_c(-cx, -cy, -cz);
        const double scale = (double)geom / strategy_pdf * mis_weight;
        for (int band = 0; band < n_bands; ++band) {
            const double cb = beta_band_val(c, band);
            const double lb = beta_band_val(l, band);
            if (!(cb > 0.0 && lb > 0.0)) continue;
            ++acc.overlapping_bands;
            const double cf = endpoint_response(c, band, c_to_l,
                                                 ci == 0 && c.stream == BDPT_SIDE_SENSOR);
            const double lf = endpoint_response(l, band, l_to_c);
            if (cf > 0.0) ++acc.camera_response_bands;
            if (lf > 0.0) ++acc.light_response_bands;
            const double gt = glass_t[static_cast<size_t>(band)];
            const double value = cb * lb * cf * lf * gt * scale;
            if (!(value > 0.0) || !std::isfinite(value)) continue;
            ++acc.positive_bands;
            double wr=0.0, wg=0.0, wb=0.0;
            band_to_display_rgb(band, n_bands, t5_freq_hz, wr, wg, wb);
            out_r += value * wr;
            out_g += value * wg;
            out_b += value * wb;
        }
    }

    /* Same normalized 2x2 sensor tent as the GPU shader. */
    void splat_sensor_tent(const BdptVertexRecord& c, double r, double g, double b,
                           T5ThreadAccum& acc) const
    {
        if (!(r + g + b > 0.0)) return;
        const double fy = (c.sensor_origin_y + ps->sensor_half_w) * inv_w - 0.5;
        const double fz = (c.sensor_origin_z + ps->sensor_half_h) * inv_h - 0.5;
        const int y0 = (int)std::floor(fy), z0 = (int)std::floor(fz);
        double wsum = 0.0;
        for (int dy=0; dy<=1; ++dy) for (int dz=0; dz<=1; ++dz) {
            const int iy=y0+dy, iz=z0+dz;
            if (iy<0 || iy>=res || iz<0 || iz>=res) continue;
            wsum += std::max(0.0,1.0-std::abs((double)iy-fy))
                  * std::max(0.0,1.0-std::abs((double)iz-fz));
        }
        if (!(wsum > 0.0)) return;
        for (int dy=0; dy<=1; ++dy) for (int dz=0; dz<=1; ++dz) {
            const int iy=y0+dy, iz=z0+dz;
            if (iy<0 || iy>=res || iz<0 || iz>=res) continue;
            const double w=(std::max(0.0,1.0-std::abs((double)iy-fy))
                           *std::max(0.0,1.0-std::abs((double)iz-fz)))/wsum;
            const size_t px=(size_t)iy*(size_t)res+(size_t)iz;
            acc.sensor_r[px]+=r*w; acc.sensor_g[px]+=g*w; acc.sensor_b[px]+=b*w;
        }
    }
};

/* ═════════════════════════════════════════════════════════════════════════
 * Strategy dispatch functions
 * ═════════════════════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────────────────────────────────
 * run_t5_allpairs  —  FULL_SEARCH: test every cam × light subpath pair.
 *
 * Partitions cam_items into nw chunks and submits each chunk to the pool.
 * Each chunk worker uses accums[t] for its local pixel/connection buffers.
 * ───────────────────────────────────────────────────────────────────────── */
inline void run_t5_allpairs(
    T5ConnContext&                              ctx,
    const std::vector<const BdptSubpathView*>& cam_items,
    const std::vector<const BdptSubpathView*>& light_items,
    std::vector<T5ThreadAccum>&                accums,
    ThreadPool&                                pool)
{
    const size_t nw    = accums.size();
    const size_t n_cam = cam_items.size();
    if (n_cam == 0 || nw == 0) return;

    ctx.hb_total = n_cam;
    ctx.hb_done.store(0, std::memory_order_relaxed);

    /* Heartbeat: only reads the atomic hb_done progress counter while workers
     * are live.  Reading sensor_r/g/b or exact_count while workers write them
     * is a data race — those are only safe to read after futures join. */
    std::atomic<bool> hb_stop{false};
    auto hb_fut = std::async(std::launch::async, [&]() {
        using namespace std::chrono_literals;
        while (!hb_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(1s);
            if (hb_stop.load(std::memory_order_relaxed)) break;
            const size_t done  = ctx.hb_done.load(std::memory_order_relaxed);
            const size_t total = ctx.hb_total;
            fprintf(stderr, "[T5-hb] allpairs %zu/%zu\n", done, total);
            fflush(stderr);
        }
    });

    const size_t chunk = (n_cam + nw - 1) / nw;
    std::vector<std::future<void>> futs;
    futs.reserve(nw);

    for (size_t t = 0; t < nw; ++t) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(n_cam, lo + chunk);
        if (lo >= hi) break;

        futs.push_back(pool.enqueue(
            [&ctx, &cam_items, &light_items, &accums, t, lo, hi]()
        {
            T5ThreadAccum& acc = accums[t];
            BdptCandidateScratch scratch;
            const int   r        = ctx.res;
            const float iw       = ctx.inv_w;
            const float ih       = ctx.inv_h;

            for (size_t ck = lo; ck < hi; ++ck) {
                const BdptSubpathView& cam = *cam_items[ck];
                for (const BdptSubpathView* lsp : light_items) {
                    const BdptSubpathView& light = *lsp;
                    for (size_t ci = 0; ci < cam.v.size(); ++ci) {
                        const BdptVertexRecord& c = *cam.v[ci];
                        if (!ctx.vertex_connectable(c)) continue;
                        if (!cam.prefix_valid[ci])      continue;
                        const double fy = (c.sensor_origin_y + ctx.ps->sensor_half_w) * iw - 0.5;
                        const double fz = (c.sensor_origin_z + ctx.ps->sensor_half_h) * ih - 0.5;
                        if (fy < -1.0 || fy > r || fz < -1.0 || fz > r) continue;

                        double pixel_r = 0.0, pixel_g = 0.0, pixel_b = 0.0;
                        for (size_t li = 0; li < light.v.size(); ++li)
                            ctx.try_connect_pair(cam, ci, c, light, li,
                                                  scratch, acc, pixel_r, pixel_g, pixel_b);

                        if (pixel_r + pixel_g + pixel_b > 0.0) {
                            ctx.splat_sensor_tent(c, pixel_r, pixel_g, pixel_b, acc);
                            ++acc.exact_count;
                        }
                    }
                }
                ++ctx.hb_done;
            }
            ctx.flush_connections(acc);
        }));
    }

    std::exception_ptr ep = nullptr;
    for (auto& f : futs) {
        try { f.get(); }
        catch (...) { if (!ep) ep = std::current_exception(); }
    }
    hb_stop.store(true, std::memory_order_relaxed);
    hb_fut.get();
    /* All worker futures have joined — safe to read accumulator data now. */
    {
        double energy = 0.0;
        uint64_t exact = 0, visible = 0, overlap = 0;
        uint64_t camera_response = 0, light_response = 0, positive = 0;
        for (const auto& a : accums) {
            if (!a.sensor_r.empty()) {
                energy +=
                    Eigen::Map<const Eigen::ArrayXd>(
                        a.sensor_r.data(), (Eigen::Index)a.sensor_r.size()).sum()
                  + Eigen::Map<const Eigen::ArrayXd>(
                        a.sensor_g.data(), (Eigen::Index)a.sensor_g.size()).sum()
                  + Eigen::Map<const Eigen::ArrayXd>(
                        a.sensor_b.data(), (Eigen::Index)a.sensor_b.size()).sum();
            }
            exact += a.exact_count;
            visible += a.visible_pairs;
            overlap += a.overlapping_bands;
            camera_response += a.camera_response_bands;
            light_response += a.light_response_bands;
            positive += a.positive_bands;
        }
        fprintf(stderr,
                "[T5-allpairs] done energy_rgb=%.3e visible_pairs=%llu "
                "overlap_bands=%llu camera_response=%llu light_response=%llu "
                "positive_bands=%llu legacy_exact=%llu\n",
                energy, (unsigned long long)visible,
                (unsigned long long)overlap,
                (unsigned long long)camera_response,
                (unsigned long long)light_response,
                (unsigned long long)positive,
                (unsigned long long)exact);
        fflush(stderr);
    }
    if (ep) std::rethrow_exception(ep);
}

