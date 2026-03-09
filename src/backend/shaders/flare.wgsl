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
@group(0) @binding(2) var<uniform> cinema: CinematicParams;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ghost_count = 4;
    let ghost_spacing = 0.4;
    let halo_width = 0.45;
    
    let sample_uv = vec2<f32>(1.0) - uv; // Flip UV for ghosts
    let direction = (vec2<f32>(0.5) - sample_uv) * ghost_spacing;
    
    var flare = vec3<f32>(0.0);
    
    // Ghosting
    for (var i = 0; i < ghost_count; i++) {
        let offset = direction * f32(i);
        let ghost_uv = fract(sample_uv + offset);
        let ghost_sample = textureSampleLevel(t_hdr, s_hdr, ghost_uv, 2.0).rgb;
        let brightness = max(0.0, dot(ghost_sample, vec3<f32>(0.2126, 0.7152, 0.0722)) - 1.0);
        flare += ghost_sample * brightness * (1.0 - f32(i) / f32(ghost_count));
    }
    
    // Halo
    let halo_vec = normalize(direction) * halo_width;
    let halo_uv = fract(sample_uv + halo_vec);
    let halo_sample = textureSampleLevel(t_hdr, s_hdr, halo_uv, 2.0).rgb;
    let halo_weight = smoothstep(0.0, 0.1, dot(halo_sample, vec3<f32>(0.2126, 0.7152, 0.0722)) - 1.0);
    flare += halo_sample * halo_weight * 0.5;

    return vec4<f32>(flare * cinema.bloom_intensity * 0.5, 1.0);
}
