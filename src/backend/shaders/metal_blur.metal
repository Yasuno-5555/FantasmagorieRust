#include <metal_stdlib>
using namespace metal;

struct BlurParams {
    float2 direction;
    float sigma;
    float _pad;
};

// Gaussian function
float gaussian(float x, float sigma) {
    float pi = 3.14159265359;
    return (1.0 / sqrt(2.0 * pi * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

kernel void blur_compute(
    texture2d<float, access::read> input_tex [[texture(0)]],
    texture2d<float, access::write> output_tex [[texture(1)]],
    constant BlurParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= input_tex.get_width() || gid.y >= input_tex.get_height()) {
        return;
    }

    int2 coords = int2(gid);
    int2 dims = int2(input_tex.get_width(), input_tex.get_height());

    // Kernel radius based on sigma
    int radius = int(ceil(params.sigma * 3.0));
    int clamped_radius = min(radius, 20);

    float4 color_acc = float4(0.0);
    float weight_acc = 0.0;
    int2 dir = int2(params.direction);

    for (int i = -clamped_radius; i <= clamped_radius; i++) {
        int2 offset = dir * i;
        int2 sample_coords = coords + offset;

        // Clamp to edge
        int clamped_x = clamp(sample_coords.x, 0, dims.x - 1);
        int clamped_y = clamp(sample_coords.y, 0, dims.y - 1);

        float weight = gaussian(float(i), params.sigma);
        float4 color = input_tex.read(uint2(clamped_x, clamped_y));

        color_acc += color * weight;
        weight_acc += weight;
    }

    float4 final_color = color_acc / weight_acc;
    output_tex.write(final_color, gid);
}
