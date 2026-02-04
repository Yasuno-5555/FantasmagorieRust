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

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var s_hdr: sampler;

@fragment
fn fs_bright(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_hdr, s_hdr, in.uv).rgb;
    let brightness = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0 || max(color.r, max(color.g, color.b)) > 1.0) {
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

struct BlurUniforms {
    direction: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(2) var<uniform> blur_u: BlurUniforms;

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let weight0 = 0.227027;
    let weight1 = 0.1945946;
    let weight2 = 0.1216216;
    let weight3 = 0.054054;
    let weight4 = 0.016216;
    
    let tex_offset = 1.0 / vec2<f32>(textureDimensions(t_hdr));
    var result = textureSample(t_hdr, s_hdr, in.uv).rgb * weight0;
    
    var offset = blur_u.direction * tex_offset * 1.0;
    result += textureSample(t_hdr, s_hdr, in.uv + offset).rgb * weight1;
    result += textureSample(t_hdr, s_hdr, in.uv - offset).rgb * weight1;
    
    offset = blur_u.direction * tex_offset * 2.0;
    result += textureSample(t_hdr, s_hdr, in.uv + offset).rgb * weight2;
    result += textureSample(t_hdr, s_hdr, in.uv - offset).rgb * weight2;
    
    offset = blur_u.direction * tex_offset * 3.0;
    result += textureSample(t_hdr, s_hdr, in.uv + offset).rgb * weight3;
    result += textureSample(t_hdr, s_hdr, in.uv - offset).rgb * weight3;
    
    offset = blur_u.direction * tex_offset * 4.0;
    result += textureSample(t_hdr, s_hdr, in.uv + offset).rgb * weight4;
    result += textureSample(t_hdr, s_hdr, in.uv - offset).rgb * weight4;
    
    return vec4<f32>(result, 1.0);
}
