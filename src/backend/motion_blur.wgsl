// Fantasmagorie Motion Blur Shader
// Performs directional blur along sampled velocity vector

struct CinematicParams {
    exposure: f32,
    ca_strength: f32,
    vignette_intensity: f32,
    bloom_intensity: f32,
    tonemap_mode: u32,
    bloom_mode: u32,
    grain_strength: f32,
    time: f32,
    lut_intensity: f32,
    blur_radius: f32,
    motion_blur_strength: f32,
    _pad: f32,
};

@group(0) @binding(0) var u_hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var u_vel_texture: texture_2d<f32>;
@group(0) @binding(2) var u_sampler: sampler;
@group(0) @binding(3) var<uniform> cinema: CinematicParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    // Full screen triangle
    let x = f32(i32(vid == 2u) * 4 - 1);
    let y = f32(i32(vid == 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // Sample velocity [pixels/frame]
    let velocity_sample = textureSample(u_vel_texture, u_sampler, uv).xy;
    
    // Convert velocity to UV space
    let dims = vec2<f32>(textureDimensions(u_hdr_texture));
    let velocity_uv = velocity_sample / dims * cinema.motion_blur_strength;
    
    // Early exit for static regions
    if (length(velocity_uv) < 0.0001) {
        return textureSample(u_hdr_texture, u_sampler, uv);
    }
    
    // Multi-tap directional blur
    let samples = 12;
    var color = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // We blur in the reverse direction of movement
    for (var i = 0; i < samples; i++) {
        let t = f32(i) / f32(samples - 1);
        let offset_uv = uv - velocity_uv * t;
        let s = textureSample(u_hdr_texture, u_sampler, offset_uv).rgb;
        
        // Simple weight: might use a better kernel (e.g. box or tent)
        let weight = 1.0;
        color += s * weight;
        total_weight += weight;
    }
    
    return vec4<f32>(color / total_weight, 1.0);
}
