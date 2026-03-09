struct GlobalUniforms {
    projection: mat4x4<f32>,
    time: f32,
    _pad0: f32,
    viewport_size: vec2<f32>,
};

struct ShapeInstance {
    rect: vec4<f32>,
    radii: vec4<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    params1: vec4<f32>,
    params2: vec4<i32>,
    material: vec4<f32>,
    pbr_params: vec4<f32>,
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var<storage, read> input_instances: array<ShapeInstance>;
@group(0) @binding(2) var<storage, read_write> output_instances: array<ShapeInstance>;
@group(0) @binding(3) var<storage, read_write> indirect_args: DrawIndirectArgs;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    
    // Make sure we don't read out of bounds. The host should pass total_count or we pad the buffer.
    // For safety, we can check arrayLength if it's strictly sized.
    if (id >= arrayLength(&input_instances)) {
        return;
    }
    
    let instance = input_instances[id];
    let pos = instance.rect.xy;
    let size = instance.rect.zw;
    
    var visible = true;
    if (pos.x + size.x < 0.0 || pos.x > globals.viewport_size.x ||
        pos.y + size.y < 0.0 || pos.y > globals.viewport_size.y) {
        visible = false;
    }
    
    if (visible && size.x > 0.0 && size.y > 0.0) {
        let output_idx = atomicAdd(&indirect_args.instance_count, 1u);
        output_instances[output_idx] = instance;
    }
}
