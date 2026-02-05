// Screen Space Reflections (SSR) Shader
// Linear Raymarching in Screen Space

struct GlobalUniforms {
    projection: mat4x4<f32>,
    time: f32,
    viewport_size: vec2<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> glob: GlobalUniforms;
@group(0) @binding(1) var t_hdr: texture_2d<f32>;
@group(0) @binding(2) var t_depth: texture_depth_2d;
@group(0) @binding(3) var t_aux: texture_2d<f32>; // nx, ny, roughness, reflectivity
@group(0) @binding(4) var s_linear: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(in_vertex_index) % 2) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(0) @binding(5) var t_history: texture_2d<f32>;
@group(0) @binding(6) var t_velocity: texture_2d<f32>;

const MAX_STEPS: i32 = 64;
const STEP_SIZE: f32 = 0.01;
const THICKNESS: f32 = 0.02;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let aux = textureSample(t_aux, s_linear, uv);
    let reflectivity = aux.w;
    let roughness = aux.z;
    
    // Default result handling
    var result_color = vec3<f32>(0.0);
    var result_alpha = 0.0; // Use alpha to indicate hit confidence for accumulation?
    // Actually, we store color in RGB and maybe confidence in A? 
    // For now, let's keep A=1.0 for valid reflection, A=0.0 for invalid.

    if (reflectivity > 0.0) {
        // Normal in screen space
        let normal = vec3<f32>(aux.xy * 2.0 - 1.0, 1.0);
        let view_dir = vec3<f32>(0.0, 0.0, -1.0);
        let reflect_dir = reflect(view_dir, normalize(normal));
        
        // --- Raymarching ---
        var ray_pos = vec3<f32>(uv, textureSample(t_depth, s_linear, uv));
        let ray_step = reflect_dir * STEP_SIZE;
        
        var hit = false;
        
        for (var i = 0; i < MAX_STEPS; i++) {
            ray_pos += ray_step;
            
            if (ray_pos.x < 0.0 || ray_pos.x > 1.0 || ray_pos.y < 0.0 || ray_pos.y > 1.0 || ray_pos.z < 0.0 || ray_pos.z > 1.0) {
                break;
            }
            
            let sampled_depth = textureSample(t_depth, s_linear, ray_pos.xy);
            let depth_diff = ray_pos.z - sampled_depth;
            
            if (depth_diff > 0.0 && depth_diff < THICKNESS) {
                result_color = textureSample(t_hdr, s_linear, ray_pos.xy).rgb;
                // Fresnel
                let fresnel = mix(0.1, 1.0, pow(1.0 - max(dot(normal, -view_dir), 0.0), 5.0));
                result_color *= reflectivity * fresnel;
                hit = true;
                break;
            }
        }
    }
    
    // --- Temporal Reprojection ---
    let velocity = textureSample(t_velocity, s_linear, uv).xy;
    let history_uv = uv - velocity;
    
    // Bounds check
    if (history_uv.x >= 0.0 && history_uv.x <= 1.0 && history_uv.y >= 0.0 && history_uv.y <= 1.0) {
        let history_sample = textureSample(t_history, s_linear, history_uv);
        let history_color = history_sample.rgb;
        
        // Simple blend factor (0.9 history, 0.1 current)
        // If current pixel didn't hit anything, rely MORE on history if available?
        // Or if current didn't hit, maybe it's occlusion?
        let blend = 0.9;
        result_color = mix(result_color, history_color, blend);
    }
    
    return vec4<f32>(result_color, 1.0);
}
