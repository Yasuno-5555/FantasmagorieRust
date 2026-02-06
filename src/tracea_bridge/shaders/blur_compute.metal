#include <metal_stdlib>
using namespace metal;

// Blur parameters passed from CPU
struct BlurParams {
    uint radius;
    float sigma;
    uint width;
    uint height;
};

// Compute Gaussian weight
float gaussian_weight(int offset, float sigma) {
    return exp(-0.5 * float(offset * offset) / (sigma * sigma));
}

// Horizontal blur pass (separable Gaussian)
kernel void blur_horizontal(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BlurParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 sum = float4(0.0);
    float weight_sum = 0.0;
    int radius = int(params.radius);
    
    // Sample horizontally with Gaussian weights
    for (int i = -radius; i <= radius; i++) {
        int x = int(gid.x) + i;
        if (x >= 0 && x < int(params.width)) {
            float weight = gaussian_weight(i, params.sigma);
            sum += input.read(uint2(x, gid.y)) * weight;
            weight_sum += weight;
        }
    }
    
    output.write(sum / weight_sum, gid);
}

// Vertical blur pass (separable Gaussian)
kernel void blur_vertical(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BlurParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 sum = float4(0.0);
    float weight_sum = 0.0;
    int radius = int(params.radius);
    
    // Sample vertically with Gaussian weights
    for (int i = -radius; i <= radius; i++) {
        int y = int(gid.y) + i;
        if (y >= 0 && y < int(params.height)) {
            float weight = gaussian_weight(i, params.sigma);
            sum += input.read(uint2(gid.x, y)) * weight;
            weight_sum += weight;
        }
    }
    
    output.write(sum / weight_sum, gid);
}

// Optimized blur with shared memory (for larger kernels)
// Uses threadgroup memory to reduce global memory reads
kernel void blur_horizontal_tiled(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant BlurParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Tile size with apron for the blur radius
    constexpr int TILE_SIZE = 16;
    constexpr int MAX_RADIUS = 32;
    threadgroup float4 tile[TILE_SIZE][TILE_SIZE + 2 * MAX_RADIUS];
    
    int radius = min(int(params.radius), MAX_RADIUS);
    
    // Load tile with apron
    int base_x = int(tgid.x * TILE_SIZE) - radius;
    int tile_width = TILE_SIZE + 2 * radius;
    
    for (int i = int(lid.x); i < tile_width; i += TILE_SIZE) {
        int x = clamp(base_x + i, 0, int(params.width) - 1);
        int y = clamp(int(gid.y), 0, int(params.height) - 1);
        tile[lid.y][i] = input.read(uint2(x, y));
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    // Compute blur from shared memory
    float4 sum = float4(0.0);
    float weight_sum = 0.0;
    
    for (int i = -radius; i <= radius; i++) {
        float weight = gaussian_weight(i, params.sigma);
        sum += tile[lid.y][int(lid.x) + radius + i] * weight;
        weight_sum += weight;
    }
    
    output.write(sum / weight_sum, gid);
}

// Bright pass for bloom extraction
kernel void bright_pass(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float &threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 color = input.read(gid);
    float luminance = dot(color.rgb, float3(0.299, 0.587, 0.114));
    
    // Soft threshold with knee
    float soft = luminance - threshold + 0.5;
    soft = clamp(soft, 0.0, 1.0);
    float contribution = soft * soft * (3.0 - 2.0 * soft);
    
    output.write(color * contribution, gid);
}

// Additive blend for bloom composition
kernel void bloom_composite(
    texture2d<float, access::read> base [[texture(0)]],
    texture2d<float, access::read> bloom [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant float &intensity [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 base_color = base.read(gid);
    float4 bloom_color = bloom.read(gid);
    
    output.write(base_color + bloom_color * intensity, gid);
}
