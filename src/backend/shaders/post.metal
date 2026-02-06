#include <metal_stdlib>
using namespace metal;

struct CinematicParams {
    float exposure;
    float ca_strength;
    float vignette_intensity;
    float bloom_intensity;
    uint tonemap_mode;
    uint bloom_mode;
    float grain_strength;
    float time;
    float lut_intensity;
    float blur_radius;
    float motion_blur_strength;
    uint debug_mode;
    float2 light_pos;
    float gi_intensity;
    float volumetric_intensity;
    float4 light_color;
};

struct VertexOutput {
    float4 position [[position]];
    float2 uv;
};

// Generic Vertex Shader for full screen triangles
vertex VertexOutput vs_post(uint vid [[vertex_id]]) {
    VertexOutput out;
    float x = (float)(vid / 2) * 4.0 - 1.0;
    float y = (float)(vid % 2) * 4.0 - 1.0;
    out.position = float4(x, y, 0.0, 1.0);
    out.uv = float2(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

// --- Bloom Bright Pass ---
fragment float4 fs_bright(
    VertexOutput in [[stage_in]],
    texture2d<float> t_hdr [[texture(0)]],
    sampler s_hdr [[sampler(0)]]
) {
    float3 color = t_hdr.sample(s_hdr, in.uv).rgb;
    float brightness = dot(color, float3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0 || max(color.r, max(color.g, color.b)) > 1.0) {
        return float4(color, 1.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

// --- Bloom Blur Pass ---
struct BlurUniforms {
    float2 direction;
    float2 _pad;
};

fragment float4 fs_blur(
    VertexOutput in [[stage_in]],
    texture2d<float> t_hdr [[texture(0)]],
    sampler s_hdr [[sampler(0)]],
    constant BlurUniforms& blur_u [[buffer(0)]]
) {
    const float weight[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    
    float2 tex_offset = 1.0 / float2(t_hdr.get_width(), t_hdr.get_height());
    float3 result = t_hdr.sample(s_hdr, in.uv).rgb * weight[0];
    
    for(int i = 1; i < 5; ++i) {
        float2 offset = blur_u.direction * tex_offset * float(i);
        result += t_hdr.sample(s_hdr, in.uv + offset).rgb * weight[i];
        result += t_hdr.sample(s_hdr, in.uv - offset).rgb * weight[i];
    }
    
    return float4(result, 1.0);
}

// --- Resolve Pass (ACES, Bloom Mix, Grain, etc.) ---
float3 aces_approx(float3 v) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
}

fragment float4 fs_resolve(
    VertexOutput in [[stage_in]],
    texture2d<float> t_hdr [[texture(0)]],
    sampler s_hdr [[sampler(0)]],
    texture2d<float> t_bloom [[texture(1)]],
    constant CinematicParams& cinema [[buffer(0)]]
) {
    float2 uv_centered = in.uv - 0.5;
    float r2 = dot(uv_centered, uv_centered);
    
    // Chromatic Aberration
    float ca_amount = cinema.ca_strength * r2 * 2.0;
    float3 hdr_color;
    hdr_color.r = t_hdr.sample(s_hdr, in.uv - float2(ca_amount, 0)).r;
    hdr_color.g = t_hdr.sample(s_hdr, in.uv).g;
    hdr_color.b = t_hdr.sample(s_hdr, in.uv + float2(ca_amount, 0)).b;
    
    float3 bloom_color = t_bloom.sample(s_hdr, in.uv).rgb;
    
    // Combine
    float3 combined = hdr_color + bloom_color * cinema.bloom_intensity;
    
    // Exposure
    float3 exposed = combined * cinema.exposure;
    
    // Tone Mapping
    float3 tonemapped = (cinema.tonemap_mode == 1) ? aces_approx(exposed) : exposed;
    
    // Film Grain
    float noise = fract(sin(dot(in.uv * (cinema.time + 1.0), float2(12.9898, 78.233))) * 43758.5453);
    float3 grainy = tonemapped + (noise - 0.5) * cinema.grain_strength;
    
    // Vignette
    float vign = 1.0 - smoothstep(0.4, 1.4, length(uv_centered) * (1.0 + cinema.vignette_intensity));
    float3 final_color = grainy * vign;

    // Gamma Correction
    return float4(pow(max(final_color, 0.0), 1.0 / 2.2), 1.0);
}
