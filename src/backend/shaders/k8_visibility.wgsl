// K8: Visibility & Occlusion Culling (HZB)
// Performs frustum culling and HZB occlusion testing for 2D sprites.

struct DrawUniforms {
    rect: vec4<f32>,
    radii: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    offset: vec2<f32>,
    scale: f32,
    border_width: f32,
    elevation: f32,
    glow_strength: f32,
    lut_intensity: f32,
    mode: u32,
    is_squircle: u32,
    time: f32,
    _pad: f32,
    _pad2: f32,
};

struct CullingUniforms {
    view_proj: mat4x4<f32>,
    num_instances: u32,
    hzb_mip_levels: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CullingUniforms;
@group(0) @binding(1) var<storage, read> instances: array<DrawUniforms>;
@group(0) @binding(2) var t_hzb: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> visible_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> visible_counter: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.num_instances) {
        return;
    }

    let inst = instances[idx];
    
    // 1. Frustum Culling
    // Project AABB to clip space
    // inst.rect is [x, y, w, h]
    let p_min = uniforms.view_proj * vec4<f32>(inst.rect.xy, 0.0, 1.0);
    let p_max = uniforms.view_proj * vec4<f32>(inst.rect.xy + inst.rect.zw, 0.0, 1.0);
    
    // Simplistic clip check
    if (p_max.x < -1.1 || p_min.x > 1.1 || p_max.y < -1.1 || p_min.y > 1.1) {
        // Return but padded slightly for safety with glow/shadow
        return;
    }

    // 2. HZB Occlusion Culling (Currently bypassed/stubbed to ensure basic visibility)
    // sample max depth etc...
    
    // 3. Mark as Visible
    let out_idx = atomicAdd(&visible_counter, 1u);
    visible_indices[out_idx] = idx;
}
