struct GlobalUniforms {
    projection: mat4x4<f32>,
    time: f32,
    _pad0: f32,
    resolution: vec2<f32>,
};

struct ShapeInstance {
    rect: vec4<f32>,
    radii: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    params1: vec4<f32>, // border_width, elevation, glow_strength, lut_intensity
    params2: vec4<i32>, // mode, is_squircle, _r1, _r2
    material: vec4<f32>, // velocity_x, velocity_y, reflectivity, roughness
    pbr_params: vec4<f32>, // normal_map_id, distortion, emissive_intensity, parallax_factor
};

// --- Instanced Bindings (Layout: 0=Global, 1=Instances, 2=Font, 3=Backdrop, 4=Sampler) ---
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;
@group(0) @binding(2) var t_diffuse: texture_2d<f32>;
@group(0) @binding(3) var t_backdrop: texture_2d<f32>;
@group(0) @binding(4) var s_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) instance_index: u32,
};

@vertex
fn vs_main(
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = globals.projection * vec4<f32>(input.position, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    out.instance_index = 0xFFFFFFFFu;
    return out;
}

@vertex
fn vs_instanced(
    input: VertexInput,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = instances[instance_index];
    var out: VertexOutput;
    let world_pos = vec2<f32>(
        instance.rect.x + input.position.x * instance.rect.z,
        instance.rect.y + input.position.y * instance.rect.w
    );
    out.clip_position = globals.projection * vec4<f32>(world_pos, 0.0, 1.0);
    out.uv = input.uv;
    out.color = instance.color;
    out.instance_index = instance_index;
    return out;
}

struct GBufferOutput {
    @location(0) color: vec4<f32>,
    @location(1) aux: vec4<f32>,
    @location(2) velocity: vec4<f32>,
    @location(3) extra: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn fs_instanced(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

@fragment
fn fs_instanced_gbuffer(in: VertexOutput) -> GBufferOutput {
    let instance = instances[in.instance_index];
    var out: GBufferOutput;
    out.color = in.color;
    // Aux: normal_x, normal_y, roughness, reflectivity
    // Assuming default normal map for now [0.5, 0.5, 1.0] -> [0.5, 0.5]
    out.aux = vec4<f32>(0.5, 0.5, instance.material.w, instance.material.z);
    out.velocity = vec4<f32>(instance.material.x, instance.material.y, 0.0, 1.0);
    // Extra: normal_map_id, distortion, emissive_intensity, parallax_factor
    out.extra = instance.pbr_params;
    return out;
}
