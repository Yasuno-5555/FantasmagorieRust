// Port of flare.wgsl
// Note: This file is concatenated with metal_shader.metal, so CinematicParams, ResolveVertexOut, etc. are already defined

fragment float4 fs_flare(ResolveVertexOut in [[stage_in]],
                       texture2d<float> t_hdr [[texture(0)]],
                       sampler s_hdr [[sampler(0)]],
                       constant CinematicParams& cinema [[buffer(0)]])
{
    float2 uv = in.uv;
    const int ghost_count = 4;
    const float ghost_spacing = 0.4;
    const float halo_width = 0.45;
    
    float2 sample_uv = float2(1.0) - uv; // Flip UV for ghosts
    float2 direction = (float2(0.5) - sample_uv) * ghost_spacing;
    
    float3 flare = float3(0.0);
    
    // Ghosting
    for (int i = 0; i < ghost_count; i++) {
        float2 offset = direction * float(i);
        float2 ghost_uv = fract(sample_uv + offset);
        float3 ghost_sample = t_hdr.sample(s_hdr, ghost_uv, level(2.0)).rgb;
        float brightness = max(0.0, dot(ghost_sample, float3(0.2126, 0.7152, 0.0722)) - 1.0);
        flare += ghost_sample * brightness * (1.0 - float(i) / float(ghost_count));
    }
    
    // Halo
    float2 halo_vec = normalize(direction) * halo_width;
    float2 halo_uv = fract(sample_uv + halo_vec);
    float3 halo_sample = t_hdr.sample(s_hdr, halo_uv, level(2.0)).rgb;
    float halo_weight = smoothstep(0.0, 0.1, dot(halo_sample, float3(0.2126, 0.7152, 0.0722)) - 1.0);
    flare += halo_sample * halo_weight * 0.5;

    return float4(flare * cinema.bloom_intensity * 0.5, 1.0);
}
