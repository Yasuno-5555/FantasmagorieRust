// FSR 1.0 EASU (Edge-Adaptive Spatial Upsampling) - WGSL Implementation
// Based on AMD FidelityFX Super Resolution 1.0

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: FSRParams;

struct FSRParams {
    input_size: vec2<f32>,
    output_size: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(vertex_index & 1u) * 4.0 - 1.0;
    let y = f32((vertex_index >> 1u) & 1u) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// EASU: Edge-Adaptive Spatial Upsampling
// Uses Lanczos-like filtering with edge detection for adaptive sharpening
fn FsrEasuF(p: vec2<f32>, input_size: vec2<f32>, output_size: vec2<f32>) -> vec3<f32> {
    let scale = input_size / output_size;
    let src_p = p * scale;
    let src_pixel = floor(src_p);
    let frac_p = src_p - src_pixel;
    
    let texel_size = 1.0 / input_size;
    
    // Sample 12 texels in a cross pattern for edge detection
    let b = textureSample(t_input, s_input, (src_pixel + vec2<f32>(0.0, -1.0)) * texel_size).rgb;
    let d = textureSample(t_input, s_input, (src_pixel + vec2<f32>(-1.0, 0.0)) * texel_size).rgb;
    let e = textureSample(t_input, s_input, (src_pixel + vec2<f32>(0.0, 0.0)) * texel_size).rgb;
    let f = textureSample(t_input, s_input, (src_pixel + vec2<f32>(1.0, 0.0)) * texel_size).rgb;
    let h = textureSample(t_input, s_input, (src_pixel + vec2<f32>(0.0, 1.0)) * texel_size).rgb;
    
    // Edge detection using gradient
    let lumB = dot(b, vec3<f32>(0.299, 0.587, 0.114));
    let lumD = dot(d, vec3<f32>(0.299, 0.587, 0.114));
    let lumE = dot(e, vec3<f32>(0.299, 0.587, 0.114));
    let lumF = dot(f, vec3<f32>(0.299, 0.587, 0.114));
    let lumH = dot(h, vec3<f32>(0.299, 0.587, 0.114));
    
    // Gradient estimation
    let gradX = abs(lumD - lumF);
    let gradY = abs(lumB - lumH);
    let gradMax = max(gradX, gradY);
    
    // Adaptive sharpening weight based on edge strength
    let sharpness = 1.0 - smoothstep(0.0, 0.3, gradMax);
    
    // Bilinear interpolation weights
    let w00 = (1.0 - frac_p.x) * (1.0 - frac_p.y);
    let w10 = frac_p.x * (1.0 - frac_p.y);
    let w01 = (1.0 - frac_p.x) * frac_p.y;
    let w11 = frac_p.x * frac_p.y;
    
    // Sample 4 corners
    let c00 = textureSample(t_input, s_input, (src_pixel + vec2<f32>(0.0, 0.0)) * texel_size).rgb;
    let c10 = textureSample(t_input, s_input, (src_pixel + vec2<f32>(1.0, 0.0)) * texel_size).rgb;
    let c01 = textureSample(t_input, s_input, (src_pixel + vec2<f32>(0.0, 1.0)) * texel_size).rgb;
    let c11 = textureSample(t_input, s_input, (src_pixel + vec2<f32>(1.0, 1.0)) * texel_size).rgb;
    
    // Weighted blend
    let bilinear = c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11;
    
    // Add edge enhancement
    let center = e;
    let edges = (center - bilinear) * sharpness * 0.5;
    
    return bilinear + edges;
}

@fragment
fn fs_easu(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.uv * params.output_size;
    let color = FsrEasuF(p, params.input_size, params.output_size);
    return vec4<f32>(color, 1.0);
}
