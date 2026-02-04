@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var s_hdr: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(4) var t_lut: texture_3d<f32>; // 3D LUT

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
    _pad: vec3<f32>,
};
@group(0) @binding(3) var<uniform> cinema: CinematicParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(in_vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
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

fn reinhard(v: vec3<f32>) -> vec3<f32> {
    return v / (1.0 + v);
}

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let dist_from_center = uv - 0.5;
    
    // 1. Chromatic Aberration
    var color: vec3<f32>;
    let ca = cinema.ca_strength;
    color.r = textureSample(t_hdr, s_hdr, uv + dist_from_center * ca).r;
    color.g = textureSample(t_hdr, s_hdr, uv).g;
    color.b = textureSample(t_hdr, s_hdr, uv - dist_from_center * ca).b;
    
    // 2. Exposure
    color *= cinema.exposure;

    // 3. Add Bloom
    if (cinema.bloom_mode > 0u) {
        let bloom = textureSample(t_bloom, s_hdr, uv).rgb;
        color += bloom * cinema.bloom_intensity;
    }
    
    // 4. Tone Mapping
    if (cinema.tonemap_mode == 1u) {
        color = aces_approx(color);
    } else if (cinema.tonemap_mode == 2u) {
        color = reinhard(color);
    }
    
    // 5. Vignette
    let vignette = 1.0 - dot(dist_from_center, dist_from_center) * 1.5;
    color *= max(vignette, cinema.vignette_intensity);
    
    // 6. Film Grain
    let noise = rand(uv + fract(cinema.time)); 
    color += (noise - 0.5) * cinema.grain_strength;
    
    // 7. Dithering (to prevent banding)
    let dither = rand(uv * 10.0) / 255.0;
    color += dither;
    
    // 8. LUT placeholder (Expand when 3D textures are ready)
    // if (cinema.lut_intensity > 0.0) {
    //    let lut_color = textureSample(t_lut, s_hdr, color).rgb;
    //    color = mix(color, lut_color, cinema.lut_intensity);
    // }

    return vec4<f32>(color, 1.0);
}
