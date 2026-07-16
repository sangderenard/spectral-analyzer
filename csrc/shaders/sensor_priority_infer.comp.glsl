#version 430 core

/* Exact runtime counterpart of SensorWorkValueNet:
 * normalized sensor RGB/exposure -> conv3x3(4,8) -> ReLU -> conv1x1 -> softplus. */
layout(local_size_x = 8, local_size_y = 8) in;

const int HIDDEN = 8;
const int INPUTS = 4;
const int CONV_WEIGHTS = HIDDEN * INPUTS * 9;
const int CONV_BIAS = CONV_WEIGHTS;
const int HEAD_WEIGHTS = CONV_BIAS + HIDDEN;
const int HEAD_BIAS = HEAD_WEIGHTS + HIDDEN;

layout(std430, binding = 0) readonly buffer SensorRgbBuf { uint sensor_rgb[]; };
layout(std430, binding = 1) readonly buffer SensorWeightBuf { uint sensor_weight[]; };
layout(std430, binding = 2) writeonly buffer LearnedMapBuf { float learned_map[]; };
layout(std430, binding = 3) readonly buffer NetworkParamBuf { float network_params[]; };

uniform uint sensor_res;

vec4 input_features(ivec2 point) {
    point = clamp(point, ivec2(0), ivec2(int(sensor_res) - 1));
    uint pixel = uint(point.x) * sensor_res + uint(point.y);
    uint plane = sensor_res * sensor_res;
    float exposure = uintBitsToFloat(sensor_weight[pixel]);
    vec3 rgb = vec3(
        uintBitsToFloat(sensor_rgb[pixel]),
        uintBitsToFloat(sensor_rgb[pixel + plane]),
        uintBitsToFloat(sensor_rgb[pixel + 2u * plane]));
    rgb = exposure > 0.0 ? max(rgb / exposure, vec3(0.0)) : vec3(0.0);
    float total = rgb.r + rgb.g + rgb.b;
    float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    return vec4(
        luma / (luma + 0.05),
        rgb.r / (total + 1.0e-6),
        rgb.g / (total + 1.0e-6),
        clamp(log(1.0 + max(exposure, 0.0)) / log(65.0), 0.0, 1.0));
}

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= sensor_res || gid.y >= sensor_res) return;
    ivec2 center = ivec2(gid);
    float hidden[HIDDEN];
    for (int h = 0; h < HIDDEN; ++h) {
        float value = network_params[CONV_BIAS + h];
        for (int c = 0; c < INPUTS; ++c) {
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    vec4 features = input_features(center + ivec2(kx - 1, ky - 1));
                    int index = (((h * INPUTS + c) * 3 + ky) * 3 + kx);
                    value += features[c] * network_params[index];
                }
            }
        }
        hidden[h] = max(value, 0.0);
    }
    float result = network_params[HEAD_BIAS];
    for (int h = 0; h < HIDDEN; ++h)
        result += hidden[h] * network_params[HEAD_WEIGHTS + h];
    float positive = result > 20.0 ? result
        : (result < -20.0 ? exp(result) : log(1.0 + exp(result)));
    learned_map[gid.x * sensor_res + gid.y] = max(positive, 0.0);
}
