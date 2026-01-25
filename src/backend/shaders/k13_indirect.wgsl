// K13: Indirect Dispatch Tower (Vulkan Compute Shader via WGSL)
// Generates VkDrawIndirectCommand and VkDispatchIndirectCommand

struct DrawIndirectCommand {
    vertexCount: u32,
    instanceCount: u32,
    firstVertex: u32,
    firstInstance: u32,
};

struct DispatchIndirectCommand {
    x: u32,
    y: u32,
    z: u32,
};

@group(0) @binding(0) var<storage, read> counters: array<u32>;
@group(0) @binding(1) var<storage, read_write> draw_cmds: array<DrawIndirectCommand>;
@group(0) @binding(2) var<storage, read_write> dispatch_cmds: array<DispatchIndirectCommand>;

@compute @workgroup_size(1)
fn main() {
    let visible_sprites = counters[0];
    let active_particles = counters[1];
    let sdf_updates = counters[2];

    // K1: Sprite Draw
    draw_cmds[0].vertexCount = 6u;
    draw_cmds[0].instanceCount = visible_sprites;
    draw_cmds[0].firstVertex = 0u;
    draw_cmds[0].firstInstance = 0u;

    // K6: Particle Draw
    draw_cmds[1].vertexCount = 6u;
    draw_cmds[1].instanceCount = active_particles;
    draw_cmds[1].firstVertex = 0u;
    draw_cmds[1].firstInstance = 0u;

    // K8: Visibility Dispatch
    dispatch_cmds[0].x = (visible_sprites + 63u) / 64u;
    dispatch_cmds[0].y = 1u;
    dispatch_cmds[0].z = 1u;

    // K5: SDF JFA Dispatch
    dispatch_cmds[1].x = (sdf_updates + 255u) / 256u;
    dispatch_cmds[1].y = 1u;
    dispatch_cmds[1].z = 1u;
}
