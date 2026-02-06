#include <metal_stdlib>
using namespace metal;

// JFA parameters
struct JFAParams {
    uint step_size;
    uint width;
    uint height;
    float max_distance;
};

// Invalid coordinate marker
constant float2 INVALID = float2(-1.0, -1.0);

// Check if coordinate is valid
bool is_valid(float2 coord) {
    return coord.x >= 0.0 && coord.y >= 0.0;
}

// Seed pass: Initialize JFA with seed positions
// Non-zero pixels become seeds (store their own position)
// Zero pixels are marked as invalid
kernel void jfa_seed(
    texture2d<float, access::read> seed_tex [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant JFAParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 seed = seed_tex.read(gid);
    
    // If any channel is non-zero, this is a seed
    if (seed.r > 0.0 || seed.g > 0.0 || seed.b > 0.0 || seed.a > 0.0) {
        // Store own position as the closest seed
        output.write(float4(float(gid.x), float(gid.y), 0.0, 0.0), gid);
    } else {
        // Mark as invalid (no seed found yet)
        output.write(float4(INVALID, 0.0, 0.0), gid);
    }
}

// Flood pass: Propagate closest seed information
// Each pixel samples 8 neighbors at step_size distance
kernel void jfa_flood(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant JFAParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float2 pos = float2(gid);
    float2 best_seed = INVALID;
    float best_dist = 1e10;
    int step = int(params.step_size);
    
    // Sample 9 positions (self + 8 neighbors at step distance)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 sample_pos = int2(gid) + int2(dx, dy) * step;
            
            // Clamp to texture bounds
            sample_pos = clamp(sample_pos, int2(0), int2(params.width - 1, params.height - 1));
            
            float4 sample_data = input.read(uint2(sample_pos));
            float2 seed_pos = sample_data.xy;
            
            if (is_valid(seed_pos)) {
                float dist = length(pos - seed_pos);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_seed = seed_pos;
                }
            }
        }
    }
    
    output.write(float4(best_seed, 0.0, 0.0), gid);
}

// Resolve pass: Convert closest seed positions to distance values
kernel void jfa_resolve(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant JFAParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 data = input.read(gid);
    float2 seed_pos = data.xy;
    float2 pos = float2(gid);
    
    float distance = 0.0;
    
    if (is_valid(seed_pos)) {
        distance = length(pos - seed_pos);
        // Normalize to 0-1 range based on max_distance
        distance = clamp(distance / params.max_distance, 0.0, 1.0);
    } else {
        // No seed found, maximum distance
        distance = 1.0;
    }
    
    // Output: R = distance, G = normalized closest seed x, B = normalized closest seed y
    float2 normalized_seed = seed_pos / float2(params.width, params.height);
    output.write(float4(distance, normalized_seed.x, normalized_seed.y, 1.0), gid);
}

// Voronoi resolve: Output seed ID instead of distance
kernel void jfa_voronoi(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant JFAParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float4 data = input.read(gid);
    float2 seed_pos = data.xy;
    
    if (is_valid(seed_pos)) {
        // Encode seed position as color
        float2 normalized = seed_pos / float2(params.width, params.height);
        output.write(float4(normalized.x, normalized.y, 0.0, 1.0), gid);
    } else {
        output.write(float4(0.0, 0.0, 0.0, 0.0), gid);
    }
}

// Signed distance field: positive outside, negative inside
kernel void jfa_signed_distance(
    texture2d<float, access::read> mask [[texture(0)]],
    texture2d<float, access::read> distance_field [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant JFAParams &params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    
    float mask_val = mask.read(gid).r;
    float dist = distance_field.read(gid).r;
    
    // If inside mask, distance is negative
    float signed_dist = mask_val > 0.5 ? -dist : dist;
    
    output.write(float4(signed_dist * 0.5 + 0.5, 0.0, 0.0, 1.0), gid);
}
