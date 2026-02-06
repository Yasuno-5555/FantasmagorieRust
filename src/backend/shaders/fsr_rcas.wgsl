// FSR 1.0 RCAS (Robust Contrast-Adaptive Sharpening) - WGSL Implementation
// Based on AMD FidelityFX Super Resolution 1.0

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;
@group(0) @binding(2) var<uniform> params: RCASParams;

struct RCASParams {
    sharpness: f32,    // 0.0 = no sharpening, 1.0 = max sharpening
    _padding: vec3<f32>,
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

// RCAS: Robust Contrast-Adaptive Sharpening
fn FsrRcasF(p: vec2<f32>, sharpness: f32) -> vec3<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(t_input));
    
    // Sample 5 pixels (cross pattern)
    let b = textureSample(t_input, s_input, p + vec2<f32>(0.0, -texel_size.y)).rgb;
    let d = textureSample(t_input, s_input, p + vec2<f32>(-texel_size.x, 0.0)).rgb;
    let e = textureSample(t_input, s_input, p).rgb;
    let f = textureSample(t_input, s_input, p + vec2<f32>(texel_size.x, 0.0)).rgb;
    let h = textureSample(t_input, s_input, p + vec2<f32>(0.0, texel_size.y)).rgb;
    
    // Compute luminance
    let lumB = dot(b, vec3<f32>(0.299, 0.587, 0.114));
    let lumD = dot(d, vec3<f32>(0.299, 0.587, 0.114));
    let lumE = dot(e, vec3<f32>(0.299, 0.587, 0.114));
    let lumF = dot(f, vec3<f32>(0.299, 0.587, 0.114));
    let lumH = dot(h, vec3<f32>(0.299, 0.587, 0.114));
    
    // Compute local contrast
    let minLum = min(min(min(lumB, lumD), min(lumE, lumF)), lumH);
    let maxLum = max(max(max(lumB, lumD), max(lumE, lumF)), lumH);
    let range = maxLum - minLum;
    
    // Avoid division by zero and limit sharpening in low-contrast areas
    let rcpRange = 1.0 / max(range, 0.001);
    
    // Compute sharpening weight (stronger in high-contrast areas)
    let w = clamp(range * rcpRange * sharpness * 0.25, 0.0, 1.0);
    
    // Laplacian-like sharpening filter
    let laplacian = (b + d + f + h) * 0.25 - e;
    
    // Apply sharpening with contrast-adaptive limiting
    let sharpened = e - laplacian * w;
    
    // Clamp to prevent ringing artifacts
    let minColor = min(min(b, d), min(f, h));
    let maxColor = max(max(b, d), max(f, h));
    
    return clamp(sharpened, minColor, maxColor);
}

@fragment
fn fs_rcas(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = FsrRcasF(in.uv, params.sharpness);
    return vec4<f32>(color, 1.0);
}
