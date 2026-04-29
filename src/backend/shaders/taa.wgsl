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
    shadow_softness: f32,
    _pad1: f32,
    _pad2: vec2<f32>,
}

@group(0) @binding(0) var t_current: texture_2d<f32>;
@group(0) @binding(1) var t_history: texture_2d<f32>;
@group(0) @binding(2) var t_velocity: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var<uniform> params: CinematicParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index) << 1u & 2) - 1.0;
    let y = f32(i32(in_vertex_index) & 2) - 1.0;
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn rgb_to_ycbcr(rgb: vec3<f32>) -> vec3<f32> {
    let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    let cb = (rgb.b - y) * 0.564;
    let cr = (rgb.r - y) * 0.713;
    return vec3<f32>(y, cb, cr);
}

fn ycbcr_to_rgb(ycbcr: vec3<f32>) -> vec3<f32> {
    let r = ycbcr.x + 1.402 * ycbcr.z;
    let g = ycbcr.x - 0.344 * ycbcr.y - 0.714 * ycbcr.z;
    let b = ycbcr.x + 1.772 * ycbcr.y;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // 1. Current sample (jittered)
    let current_raw = textureSample(t_current, s_linear, uv).rgb;
    
    // 2. Reproject history
    let velocity = textureSample(t_velocity, s_linear, uv).xy;
    let history_uv = uv - velocity;
    
    // 3. Color Clamping to reduce ghosting
    // Sample neighborhood to find local min/max
    let texel = 1.0 / params.render_size;
    var c_min = current_raw;
    var c_max = current_raw;
    
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0)
    );
    
    for (var i = 0; i < 4; i++) {
        let neighbor = textureSample(t_current, s_linear, uv + offsets[i] * texel).rgb;
        c_min = min(c_min, neighbor);
        c_max = max(c_max, neighbor);
    }
    
    // 4. Sample and clamp history
    var history = textureSample(t_history, s_linear, history_uv).rgb;
    history = clamp(history, c_min, c_max);
    
    // 5. Accumulate
    // Higher blend factor for stability, lower for responsiveness
    // In 2D, we can afford high stability.
    let blend = 0.95;
    let result = mix(current_raw, history, blend);
    
    return vec4<f32>(result, 1.0);
}
