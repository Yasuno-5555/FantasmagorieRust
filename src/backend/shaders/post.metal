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
    float2 jitter;
    float2 render_size;
};

// ... (VertexOutput and vs_post stay same)

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
    
    // Chromatic Aberration
    float3 hdr_color;
    float3 bloom_color;
    if (cinema.ca_strength > 0.0) {
        float ca_amount = cinema.ca_strength * 0.01 * length(uv_centered);
        hdr_color.r = t_hdr.sample(s_hdr, in.uv + float2(ca_amount, 0)).r;
        hdr_color.g = t_hdr.sample(s_hdr, in.uv).g;
        hdr_color.b = t_hdr.sample(s_hdr, in.uv - float2(ca_amount, 0)).b;
        
        bloom_color.r = t_bloom.sample(s_hdr, in.uv + float2(ca_amount, 0)).r;
        bloom_color.g = t_bloom.sample(s_hdr, in.uv).g;
        bloom_color.b = t_bloom.sample(s_hdr, in.uv - float2(ca_amount, 0)).b;
    } else {
        hdr_color = t_hdr.sample(s_hdr, in.uv).rgb;
        bloom_color = t_bloom.sample(s_hdr, in.uv).rgb;
    }
    
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
    float3 out_color = pow(max(final_color, 0.0), 1.0 / 2.2);
    
    // Dithering to prevent banding
    float dither_noise = fract(sin(dot(in.uv, float2(12.9898, 78.233))) * 43758.5453) / 255.0;
    out_color += dither_noise;

    return float4(out_color, 1.0);
}
