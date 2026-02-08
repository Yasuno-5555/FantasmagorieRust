struct GlobalUniforms {
    projection: mat4x4<f32>,
    time: f32,
    _pad0: f32,
    resolution: vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_sampler: sampler;
@group(0) @binding(3) var<storage, read> bone_matrices: array<mat4x4<f32>>; // Global palette

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) bone_indices: vec4<u32>,
    @location(4) bone_weights: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate skinning matrix
    var skin_mat = mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0)
    );
    
    // Accumulate weighted bone matrices
    // We iterate 4 times manually because loops can be tricky in some older WGSL versions(?), 
    // but standard for loop is fine.
    
    for (var i = 0u; i < 4u; i++) {
        let bone_idx = input.bone_indices[i];
        let weight = input.bone_weights[i];
        if (weight > 0.0) {
            skin_mat += bone_matrices[bone_idx] * weight;
        }
    }
    
    // If no weights (e.g. weight sum is 0), fallback to identity? 
    // Usually weights should sum to 1.0. 
    // If total weight is too small, assume identity or static.
    // For now assume valid weights.
    
    let local_pos = vec4<f32>(input.position, 0.0, 1.0);
    // Transform position by skin matrix (which includes bone transforms)
    let world_pos = skin_mat * local_pos;
    
    out.clip_position = globals.projection * world_pos;
    out.uv = input.uv;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_sampler, in.uv) * in.color;
}

struct GBufferOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @location(2) velocity: vec4<f32>,
    @location(3) extra: vec4<f32>,
};

@fragment
fn fs_gbuffer(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;
    let color = textureSample(t_diffuse, s_sampler, in.uv) * in.color;
    if (color.a < 0.1) { discard; }
    
    out.color = color;
    // Default material for skinned meshes
    out.aux = vec4<f32>(0.5, 0.5, 0.5, 0.0); // NormalX, NormalY, Roughness, Refl
    out.velocity = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.extra = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
