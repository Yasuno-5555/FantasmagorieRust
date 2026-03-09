// Port of dof.wgsl
// Note: This file is concatenated with metal_shader.metal, so CinematicParams, ResolveVertexOut, etc. are already defined

fragment float4 fs_dof(ResolveVertexOut in [[stage_in]],
                       texture2d<float> t_hdr [[texture(0)]],
                       texture2d<float> t_depth [[texture(1)]],
                       sampler s_hdr [[sampler(0)]],
                       constant CinematicParams& cinema [[buffer(0)]])
{
    float depth = t_depth.sample(s_hdr, in.uv).r;
    
    // Simple CoC calculation
    float focal_plane = 0.5;
    float coc = abs(depth - focal_plane) * cinema.blur_radius;
    
    if (coc < 0.01) {
        return t_hdr.sample(s_hdr, in.uv);
    }

    float3 color = float3(0.0);
    float total_weight = 0.0;
    
    const int samples = 16;
    for (int i = 0; i < samples; i++) {
        float angle = float(i) * 2.39996; // Golden angle
        float radius = coc * sqrt(float(i) / float(samples));
        float2 offset = float2(cos(angle), sin(angle)) * radius / cinema.render_size;
        
        float3 sample_color = t_hdr.sample(s_hdr, in.uv + offset, level(0.0)).rgb;
        color += sample_color;
        total_weight += 1.0;
    }
    
    return float4(color / total_weight, 1.0);
}
