@group(0) @binding(0) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var aux_tex: texture_2d<f32>; // xy=Normal, z=Roughness, w=Elevation
@group(0) @binding(3) var depth_tex: texture_depth_2d; // Can bind depth as texture? Usually depth is separate from bind group in WGPU unless sampled.
@group(0) @binding(4) var sampler_linear: sampler;

struct SSRUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    resolution: vec2<f32>,
    max_steps: u32,
    stride: f32,
    thickness: f32,
    jitter: f32,
};
@group(0) @binding(5) var<uniform> u: SSRUniforms;

fn get_position_view_space(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv * 2.0 - 1.0, depth);
    // Note: WGPU/Metal use 0..1 depth? Or reverse-z? Assuming standard 0..1 for now.
    // Invert Y?
    // let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth); // If Y-up vs Y-down
    // Let's assume consistent UV.
    
    let clip_pos = vec4<f32>(ndc, 1.0);
    // let view_pos = u.inv_proj_matrix * clip_pos;
    // return view_pos.xyz / view_pos.w;
    
    // For now, simpler approximation if orthographic/perspective is known. 
    // But taking inverse proj is standard.
    
    // Implementation TBD: Check projection matrix conventions.
    return vec3<f32>(0.0); // Placeholder
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_tex);
    if (coords.x >= i32(dims.x) || coords.y >= i32(dims.y)) { return; }
    
    let uv = vec2<f32>(coords) / vec2<f32(dims.x), f32(dims.y));
    
    // 1. Sample G-Buffer
    let aux = textureLoad(aux_tex, coords, 0);
    let roughness = aux.z;
    if (roughness > 0.8) {
        textureStore(output_tex, coords, vec4<f32>(0.0));
        return;
    }
    
    let normal_encoded = aux.xy;
    let normal = normalize(vec3<f32>(normal_encoded * 2.0 - 1.0, sqrt(max(0.0, 1.0 - dot(normal_encoded * 2.0 - 1.0, normal_encoded * 2.0 - 1.0))))); 
    // Basic reconstruction from 2 components assuming z > 0
    
    // 2. Perform Raymarch (Placeholder)
    // ...
    
    // 3. Sample HDR at hit
    let final_color = textureSampleLevel(hdr_tex, sampler_linear, uv, 0.0); // TBD: Use hit UV
    
    textureStore(output_tex, coords, final_color * 0.5); // Dim reflection
}
