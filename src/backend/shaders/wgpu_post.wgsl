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
};

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var s_hdr: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var<uniform> cinema: CinematicParams;
@group(0) @binding(4) var t_lut: texture_3d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn sample_lut(color: vec3<f32>) -> vec3<f32> {
    let uvw = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSample(t_lut, s_hdr, uvw).rgb;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_centered = in.uv - 0.5;
    let r2 = dot(uv_centered, uv_centered);
    
    // Simple sampling (no distortion for now to isolate crash)
    let dist_uv = in.uv;
    
    // Sample Lighting Result
    let scene_color = textureSample(t_hdr, s_hdr, dist_uv).rgb;
    
    // Apply Bloom
    let bloom_color = textureSample(t_bloom, s_hdr, dist_uv).rgb;
    let combined = scene_color + bloom_color * cinema.bloom_intensity;
    
    // Exposure
    let exposed = combined * cinema.exposure;
    
    // Tone Mapping
    var tonemapped = exposed;
    if (cinema.tonemap_mode == 1u) {
        tonemapped = aces_approx(exposed);
    }
    
    // LUT Grading
    let lut_color = sample_lut(tonemapped);
    let graded = mix(tonemapped, lut_color, cinema.lut_intensity);
    
    // Film Grain
    let noise = fract(sin(dot(in.uv * (cinema.time + 1.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let grainy = graded + (noise - 0.5) * cinema.grain_strength;
    
    // Vignette
    let vign = 1.0 - smoothstep(0.4, 1.4, length(uv_centered) * (1.0 + cinema.vignette_intensity));
    let final_color = grainy * vign;

    // Gamma Correction
    let out_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(out_color, 1.0);
}
