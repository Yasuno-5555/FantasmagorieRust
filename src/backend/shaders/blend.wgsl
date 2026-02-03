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

@group(0) @binding(0) var t_bg: texture_2d<f32>;
@group(0) @binding(1) var s_bg: sampler;
@group(0) @binding(2) var t_fg: texture_2d<f32>;
@group(0) @binding(3) var s_fg: sampler;

struct BlendUniforms {
    opacity: f32,
    mode: u32, // 0: Alpha, 1: Additive, 2: Multiply
    _pad: array<vec4<f32>, 63>, 
};

@group(0) @binding(4) var<uniform> uniforms: BlendUniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bg = textureSample(t_bg, s_bg, in.uv);
    let fg = textureSample(t_fg, s_fg, in.uv);
    
    var result: vec4<f32>;
    
    if (uniforms.mode == 0u) {
        // Alpha Blend
        let alpha = fg.a * uniforms.opacity;
        result = vec4<f32>(mix(bg.rgb, fg.rgb, alpha), 1.0);
    } else if (uniforms.mode == 1u) {
        // Additive
        result = vec4<f32>(bg.rgb + fg.rgb * uniforms.opacity, 1.0);
    } else if (uniforms.mode == 2u) {
        // Multiply
        result = vec4<f32>(bg.rgb * mix(vec3<f32>(1.0), fg.rgb, uniforms.opacity), 1.0);
    } else {
        result = fg;
    }
    
    return result;
}
