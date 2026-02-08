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
    debug_mode: u32,
    light_pos: vec2<f32>,
    gi_intensity: f32,
    volumetric_intensity: f32,
    light_color: vec4<f32>,
    jitter: vec2<f32>,
    render_size: vec2<f32>,
}

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: CinematicParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Full screen triangle
    let x = f32(i32(in_vertex_index) << 1u & 2) - 1.0;
    let y = f32(i32(in_vertex_index) & 2) - 1.0;
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Jitter-corrected sampling
    // We subtract the jitter offset because the camera was offset by +jitter
    let uv = in.uv - params.jitter;
    
    // Bilinear sample
    var color = textureSample(t_input, s_input, uv);
    
    // Simple sharpening (Unsharp Mask style)
    let texel_size = 1.0 / params.render_size;
    let neighbor = textureSample(t_input, s_input, uv + vec2<f32>(texel_size.x, 0.0)).rgb;
    let neighbor_neg = textureSample(t_input, s_input, uv - vec2<f32>(texel_size.x, 0.0)).rgb;
    
    let sharpen_strength = 0.2;
    let center = color.rgb;
    let edge = (neighbor + neighbor_neg) * 0.5;
    
    color = vec4<f32>(center + (center - edge) * sharpen_strength, color.a);
    
    return color;
}
