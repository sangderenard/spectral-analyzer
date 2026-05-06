#include "emitter_angle_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

static inline float srgb_to_linear(float c) {
    c = std::max(0.0f, std::min(1.0f, c));
    return (c <= 0.04045f) ? c / 12.92f
                           : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

extern "C" int eak_analyze_rgba8_layers(const uint8_t* rgba_layers,
                                         int width,
                                         int height,
                                         int layers,
                                         float active_threshold,
                                         EAKMetrics* out_metrics)
{
    if (!rgba_layers || !out_metrics || width <= 0 || height <= 0 || layers <= 0) {
        return EAK_BAD_ARGUMENT;
    }
    const int wh = width * height;
    const float inv_wh = 1.0f / (float)wh;
    const float threshold = std::max(0.0f, active_threshold);

    for (int layer = 0; layer < layers; ++layer) {
        double wsum = 0.0;
        double usum = 0.0, vsum = 0.0;
        int active_count = 0;

        const uint8_t* base = rgba_layers + (size_t)layer * (size_t)wh * 4u;
        for (int y = 0; y < height; ++y) {
            float v = ((float)y + 0.5f) / (float)height;
            for (int x = 0; x < width; ++x) {
                float u = ((float)x + 0.5f) / (float)width;
                const uint8_t* p = base + ((size_t)y * (size_t)width + (size_t)x) * 4u;
                float r = srgb_to_linear((float)p[0] * (1.0f / 255.0f));
                float g = srgb_to_linear((float)p[1] * (1.0f / 255.0f));
                float b = srgb_to_linear((float)p[2] * (1.0f / 255.0f));
                float a = (float)p[3] * (1.0f / 255.0f);
                float lum = (0.2126f * r + 0.7152f * g + 0.0722f * b) * a;
                if (lum > threshold) {
                    ++active_count;
                }
                wsum += lum;
                usum += (double)u * lum;
                vsum += (double)v * lum;
            }
        }

        EAKMetrics m{};
        m.flux_scale = (float)(wsum * (double)inv_wh);
        m.active_fraction = (float)active_count * inv_wh;
        m.active_density = (m.active_fraction > 1e-8f)
                         ? (m.flux_scale / m.active_fraction) : 0.0f;

        double cu = 0.5;
        double cv = 0.5;
        if (wsum > 1e-12) {
            cu = usum / wsum;
            cv = vsum / wsum;
        }
        m.centroid_u = (float)cu;
        m.centroid_v = (float)cv;

        double c00 = 0.0, c01 = 0.0, c11 = 0.0;
        if (wsum > 1e-12) {
            for (int y = 0; y < height; ++y) {
                float v = ((float)y + 0.5f) / (float)height;
                for (int x = 0; x < width; ++x) {
                    float u = ((float)x + 0.5f) / (float)width;
                    const uint8_t* p = base + ((size_t)y * (size_t)width + (size_t)x) * 4u;
                    float r = srgb_to_linear((float)p[0] * (1.0f / 255.0f));
                    float g = srgb_to_linear((float)p[1] * (1.0f / 255.0f));
                    float b = srgb_to_linear((float)p[2] * (1.0f / 255.0f));
                    float a = (float)p[3] * (1.0f / 255.0f);
                    double lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) * a;
                    double du = (double)u - cu;
                    double dv = (double)v - cv;
                    c00 += lum * du * du;
                    c01 += lum * du * dv;
                    c11 += lum * dv * dv;
                }
            }
            c00 /= wsum; c01 /= wsum; c11 /= wsum;
        }

        double tr = c00 + c11;
        double det_term = std::sqrt(std::max(0.0, (c00 - c11) * (c00 - c11) + 4.0 * c01 * c01));
        double l0 = std::max(0.0, 0.5 * (tr + det_term));
        double l1 = std::max(0.0, 0.5 * (tr - det_term));
        double ax = c01;
        double ay = l0 - c00;
        double alen = std::sqrt(ax * ax + ay * ay);
        if (alen <= 1e-12) {
            ax = 1.0; ay = 0.0; alen = 1.0;
        }
        m.axis_u = (float)(ax / alen);
        m.axis_v = (float)(ay / alen);
        m.spread_major = (float)std::sqrt(l0);
        m.spread_minor = (float)std::sqrt(l1);

        /* Equivalent uniform spherical cap on a hemisphere:
           active_fraction = cap_area / hemisphere_area = 1 - cos(theta). */
        float af = std::max(0.0f, std::min(1.0f, m.active_fraction));
        m.cone_cos = 1.0f - af;
        m.cone_solid_angle = 2.0f * 3.14159265358979323846f * af;

        out_metrics[layer] = m;
    }
    return EAK_OK;
}
