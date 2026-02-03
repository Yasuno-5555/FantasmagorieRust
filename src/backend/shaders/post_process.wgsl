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

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct PostProcessUniforms {
    threshold: f32,
    direction: vec2<f32>, 
    intensity: f32,
    _pad: array<vec4<f32>, 62>, 
};

@group(0) @binding(2) var<uniform> uniforms: PostProcessUniforms;

@fragment
fn fs_extract(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    let brightness = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    
    if (brightness > uniforms.threshold) {
        return color;
    } else {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
}

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_offset = 1.0 / vec2<f32>(textureDimensions(t_input));
    
    var result = textureSample(t_input, s_input, in.uv) * 0.227027;
    
    let o1 = 1.0 * tex_offset * uniforms.direction;
    result = result + textureSample(t_input, s_input, in.uv + o1) * 0.1945946;
    result = result + textureSample(t_input, s_input, in.uv - o1) * 0.1945946;

    let o2 = 2.0 * tex_offset * uniforms.direction;
    result = result + textureSample(t_input, s_input, in.uv + o2) * 0.1216216;
    result = result + textureSample(t_input, s_input, in.uv - o2) * 0.1216216;

    let o3 = 3.0 * tex_offset * uniforms.direction;
    result = result + textureSample(t_input, s_input, in.uv + o3) * 0.054054;
    result = result + textureSample(t_input, s_input, in.uv - o3) * 0.054054;

    let o4 = 4.0 * tex_offset * uniforms.direction;
    result = result + textureSample(t_input, s_input, in.uv + o4) * 0.016216;
    result = result + textureSample(t_input, s_input, in.uv - o4) * 0.016216;
    
    return result;
}
@group(0) @binding(3) var t_lut: texture_3d<f32>;
@group(0) @binding(4) var s_lut: sampler;

@fragment
fn fs_grade(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_input, s_input, in.uv);
    
    // Simple ACES-like tone mapping
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let hdr = color.rgb;
    let mapped = (hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e);
    
    // LUT lookup (3D)
    // We assume the LUT is 16x16x16 or 32x32x32
    let lut_color = textureSample(t_lut, s_lut, mapped);
    
    let result = mix(mapped, lut_color.rgb, uniforms.intensity);
    return vec4<f32>(result, color.a);
}
