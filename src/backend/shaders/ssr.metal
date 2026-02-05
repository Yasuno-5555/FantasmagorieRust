#include <metal_stdlib>
using namespace metal;

struct GlobalUniforms {
    float4x4 projection;
    float time;
    float2 viewport_size;
    float _pad;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vs_ssr(uint vid [[vertex_id]]) {
    float x = (float)(vid / 2) * 4.0 - 1.0;
    float y = (float)(vid % 2) * 4.0 - 1.0;
    
    VertexOut out;
    out.position = float4(x, y, 0.0, 1.0);
    out.uv = float2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fragment float4 fs_ssr(VertexOut in [[stage_in]],
                      constant GlobalUniforms &glob [[buffer(0)]],
                      texture2d<float> t_hdr [[texture(0)]],
                      texture2d<float> t_depth [[texture(1)]],
                      texture2d<float> t_aux [[texture(2)]],
                      texture2d<float> t_history [[texture(3)]],
                      texture2d<float> t_velocity [[texture(4)]],
                      sampler s [[sampler(0)]]) {
    float2 uv = in.uv;
    float4 aux = t_aux.sample(s, uv);
    float reflectivity = aux.w;
    float roughness = aux.z;
    
    // Default result
    float3 result_color = float3(0.0);
    
    if (reflectivity > 0.0) {
        float3 normal = float3(aux.xy * 2.0 - 1.0, 1.0);
        float3 view_dir = float3(0.0, 0.0, -1.0);
        float3 reflect_dir = reflect(view_dir, normalize(normal));
        
        float3 ray_pos = float3(uv, t_depth.sample(s, uv).r);
        float3 ray_step = reflect_dir * 0.01; // STEP_SIZE
        
        bool hit = false;
        
        for (int i = 0; i < 64; i++) {
            ray_pos += ray_step;
            
            if (any(ray_pos < 0.0) || any(ray_pos > 1.0)) {
                break;
            }
            
            float sampled_depth = t_depth.sample(s, ray_pos.xy).r;
            float depth_diff = ray_pos.z - sampled_depth;
            
            if (depth_diff > 0.0 && depth_diff < 0.02) { // THICKNESS
                result_color = t_hdr.sample(s, ray_pos.xy).rgb;
                float fresnel = mix(0.1, 1.0, pow(1.0 - max(dot(normal, -view_dir), 0.0), 5.0));
                result_color *= reflectivity * fresnel;
                hit = true;
                break;
            }
        }
    }
    
    // --- Temporal Reprojection ---
    float2 velocity = t_velocity.sample(s, uv).xy;
    float2 history_uv = uv - velocity;
    
    // Bounds check and validate
    if (all(history_uv >= 0.0) && all(history_uv <= 1.0)) {
        float3 history_color = t_history.sample(s, history_uv).rgb;
        
        float blend = 0.9;
        result_color = mix(result_color, history_color, blend);
    }
    
    return float4(result_color, 1.0);
}
