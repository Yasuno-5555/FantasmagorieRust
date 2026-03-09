#include <metal_stdlib>
using namespace metal;

struct GlobalUniforms {
    float4x4 projection;
    float time;
    float _pad0;
    float2 viewport_size;
};

struct ShapeInstance {
    float4 rect;
    float4 radii;
    float4 color;
    float4 border_color;
    float4 glow_color;
    float4 params1;
    int4 params2;
    float4 material;
    float4 pbr_params;
};

struct DrawIndirectArgs {
    uint vertex_count;
    atomic_uint instance_count;
    uint first_vertex;
    uint first_instance;
};

// 2D Frustum culling simple version: check if rect intersects viewport
// For simplicity, we just use rect limits vs [0, 0] to viewport_size
kernel void cull_instances(
    constant GlobalUniforms& globals [[buffer(0)]],
    const device ShapeInstance* input_instances [[buffer(1)]],
    device ShapeInstance* output_instances [[buffer(2)]],
    device DrawIndirectArgs& indirect_args [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // We need to know total instances count. We can pass it as a uniform or assume thread limit.
    // For now, let's assume if rect.z (width) == 0, it's invalid (or pass count in pushing constants).
    // Actually, passing total count in a bind group is better. Let's add it to buffer 3 after args or buffer 4.
    // Wait, the host dispatches exact number of threads = total instances.
    
    ShapeInstance instance = input_instances[id];
    
    // Bounds check
    float2 pos = instance.rect.xy;
    float2 size = instance.rect.zw;
    
    // Viewport bounds [0, 0] to [viewport_size.x, viewport_size.y]
    // AABB intersection
    bool visible = true;
    if (pos.x + size.x < 0.0 || pos.x > globals.viewport_size.x ||
        pos.y + size.y < 0.0 || pos.y > globals.viewport_size.y) {
        visible = false;
    }
    
    if (visible && size.x > 0.0 && size.y > 0.0) {
        uint output_idx = atomic_fetch_add_explicit(&indirect_args.instance_count, 1, memory_order_relaxed);
        output_instances[output_idx] = instance;
    }
}
