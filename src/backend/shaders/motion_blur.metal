// motion_blur.metal - Motion blur post-processing shader
// Note: This file is concatenated with metal_shader.metal, so CinematicParams, ResolveVertexOut, etc. are already defined


fragment float4 fs_motion_blur(ResolveVertexOut in [[stage_in]],
                       constant CinematicParams &cinema [[buffer(1)]],
                       texture2d<float> hdr_tex [[texture(0)]],
                       texture2d<float> vel_tex [[texture(1)]],
                       sampler s [[sampler(0)]]) {
    float2 uv = in.uv;
    
    // Sample velocity [pixels/frame]
    float2 velocity_sample = vel_tex.sample(s, uv).xy;
    
    // Convert velocity to UV space
    float2 dims = float2(hdr_tex.get_width(), hdr_tex.get_height());
    float2 velocity_uv = velocity_sample / dims * cinema.motion_blur_strength;
    
    // Early exit for static regions
    if (length_squared(velocity_uv) < 0.000001) {
        return hdr_tex.sample(s, uv);
    }
    
    // Multi-tap directional blur
    const int samples = 12;
    float3 color = float3(0.0);
    float total_weight = 0.0;
    
    for (int i = 0; i < samples; i++) {
        float t = float(i) / float(samples - 1);
        float2 offset_uv = uv - velocity_uv * t;
        float3 sample_color = hdr_tex.sample(s, offset_uv).rgb;
        
        color += sample_color;
        total_weight += 1.0;
    }
    
    return float4(color / total_weight, 1.0);
}
