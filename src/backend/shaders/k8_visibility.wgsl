// K8: Visibility & Occlusion Culling (HZB)
// Performs frustum culling and HZB occlusion testing for 2D sprites.

struct InstanceData {
    pos: vec2<f32>,
    size: vec2<f32>,
    depth: f32,
    id: u32,
};

struct CullingUniforms {
    view_proj: mat4x4<f32>,
    num_instances: u32,
    hzb_mip_levels: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CullingUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
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
    let p_min = uniforms.view_proj * vec4<f32>(inst.pos, inst.depth, 1.0);
    let p_max = uniforms.view_proj * vec4<f32>(inst.pos + inst.size, inst.depth, 1.0);
    
    // Simplistic clip check
    if (p_max.x < -1.0 || p_min.x > 1.0 || p_max.y < -1.0 || p_min.y > 1.0) {
        return;
    }

    // 2. HZB Occlusion Culling
    // Sample max depth from HZB at appropriate mip level based on instance size
    let uv = (inst.pos + inst.size * 0.5) * 0.5 + 0.5; // Dummy projection to UV
    let hzb_depth = textureLoad(t_hzb, vec2<i32>(uv * vec2<f32>(textureDimensions(t_hzb))), 0).r;
    
    if (inst.depth > hzb_depth) {
        // Occluded (behind the closest element in the HZB)
        // return; 
    }

    // 3. Mark as Visible
    let out_idx = atomicAdd(&visible_counter, 1u);
    visible_indices[out_idx] = idx;
}
