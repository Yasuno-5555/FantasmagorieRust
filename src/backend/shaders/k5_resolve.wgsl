struct JFAUniforms {
    width: u32,
    height: u32,
    jfa_step: u32,
    ping_pong_idx: u32,
    intensity: f32,
    decay: f32,
    radius: f32,
    _pad: u32,
}

var<push_constant> uniforms: JFAUniforms;

@group(0) @binding(1) var t_jfa: texture_2d<f32>;
@group(0) @binding(3) var t_sdf: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= uniforms.width || y >= uniforms.height) {
        return;
    }

    let p = vec2<i32>(i32(x), i32(y));
    let seed = textureLoad(t_jfa, p, 0).xy;
    
    var dist = 10000.0;
    if (seed.x >= 0.0) {
        let diff = vec2<f32>(f32(x), f32(y)) - seed;
        dist = length(diff);
    }
    
    // Lighting: intensity / (1.0 + decay * dist^2)
    let light = uniforms.intensity / (1.0 + uniforms.decay * dist * dist);
    
    textureStore(t_sdf, p, vec4<f32>(light, 0.0, 0.0, 1.0));
}
