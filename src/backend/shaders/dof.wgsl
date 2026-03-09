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
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var s_hdr: sampler;
@group(0) @binding(3) var<uniform> cinema: CinematicParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

fn get_coc(depth: f32) -> f32 {
    let focal_plane = 0.5; // Example fixed focal plane
    let focal_length = 0.1;
    let aperture = 0.02;
    return abs(depth - focal_plane) * cinema.blur_radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(t_depth, s_hdr, in.uv);
    let coc = get_coc(depth);
    
    if (coc < 0.01) {
        return textureSample(t_hdr, s_hdr, in.uv);
    }

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;
    
    // Bokeh Disk sampling
    let samples = 16;
    for (var i = 0; i < samples; i++) {
        let angle = f32(i) * 2.39996; // Golden angle
        let radius = coc * sqrt(f32(i) / f32(samples));
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius / cinema.render_size;
        
        let sample_color = textureSampleLevel(t_hdr, s_hdr, in.uv + offset, 0.0).rgb;
        color += sample_color;
        total_weight += 1.0;
    }
    
    return vec4<f32>(color / total_weight, 1.0);
}
